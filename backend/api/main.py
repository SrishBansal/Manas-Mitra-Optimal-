import os
import sys
import logging
import asyncio
from enum import Enum
from typing import List, Dict, Any, Optional, Literal

# 1. FORCE OFFLINE FOR PYTORCH/TRANSFORMERS
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 2. IMPORT SENTENCE_TRANSFORMERS FIRST TO AVOID WINDOWS DLL CLASH
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
for env_path in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".env")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
]:
    if os.path.exists(env_path):
        load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("Warning: GEMINI_API_KEY not found in environment!")

# Initialize Gemini client (new SDK)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Append scripts folder to path for safety check import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
try:
    from safety import check_crisis
except ImportError:
    check_crisis = None

app = FastAPI(title="Manas Mitra API", description="RAG + Gemini API backend for the Manas Mitra mental health chatbot")

# CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for ChromaDB and SentenceTransformer relative to backend/api/main.py
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(current_dir, "..", "chroma_db"))
MODEL_PATH = os.path.abspath(os.path.join(current_dir, "..", "..", "all-MiniLM-L6-v2")).replace("\\", "/")

# Initialize ChromaDB client and collection
logger.info(f"Connecting to ChromaDB at: {DB_PATH}")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

logger.info(f"Loading embedding function from: {MODEL_PATH}")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_PATH,
    device="cpu"
)

collection = chroma_client.get_or_create_collection(
    name="cognitive_distortions",
    embedding_function=embedding_func
)

# Emotion model configuration
EMOTION_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
try:
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--bhadresh-savani--distilbert-base-uncased-emotion/snapshots")
    if os.path.exists(cache_dir):
        snapshots = os.listdir(cache_dir)
        if snapshots:
            EMOTION_MODEL = os.path.abspath(os.path.join(cache_dir, snapshots[0]))
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load local emotion classification model
try:
    logger.info(f"Loading emotion classification model from: {EMOTION_MODEL} on {DEVICE}...")
    is_local_emotion = os.path.exists(EMOTION_MODEL)
    emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL, local_files_only=is_local_emotion)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL, local_files_only=is_local_emotion).to(DEVICE)
    emotion_model.eval()
    logger.info("Emotion model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading emotion model: {str(e)}")
    emotion_tokenizer = None
    emotion_model = None

class Emotion(str, Enum):
    SADNESS = "sadness"
    JOY = "joy"
    LOVE = "love"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    emotion: str
    reply: str

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "architecture": "RAG + Gemini API",
        "chroma_db_path": DB_PATH,
        "embedding_model": MODEL_PATH,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "emotion_model": EMOTION_MODEL,
        "device": DEVICE
    }

def detect_emotion(text: str) -> str:
    """Detect the emotion in the given text using the local classification model."""
    if emotion_model is None or emotion_tokenizer is None:
        return "neutral"
    try:
        inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        
        predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
        emotion = emotion_model.config.id2label[predicted_class_idx]
        return emotion.lower()
    except Exception as e:
        logger.error(f"Error in emotion detection: {str(e)}")
        return "neutral"

def _call_gemini(system_instruction: str, message: str) -> str:
    """Synchronous Gemini API call — run via asyncio.to_thread."""
    if not gemini_client:
        return "I'm here to support you, but my conversational engine is currently offline. Please configure the GEMINI_API_KEY in the environment."

    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-lite"]
    last_error = None
    for model_name in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                logger.warning(f"Rate limit hit on {model_name}, trying next model...")
                last_error = e
                continue
            logger.error(f"Gemini error on {model_name}: {err_str}")
            last_error = e
            continue
    logger.error(f"All Gemini models failed: {last_error}")
    return "I'm here for you, but I'm experiencing high demand right now. Please try again in a minute."

async def generate_response(message: str, emotion: str) -> str:
    """Generate a response using RAG retrieval and Gemini API."""
    try:
        if check_crisis is not None:
            crisis_reply = check_crisis(message, locale="en-IN")
            if crisis_reply is not None:
                return crisis_reply

        # 1. RETRIEVE closest matching distortion from ChromaDB
        try:
            results = collection.query(query_texts=[message], n_results=1)
            if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
                metadata = results['metadatas'][0][0]
                distortion_name = metadata.get("distortion", "General Support")
                definition = metadata.get("definition", "No definition available.")
                framework = metadata.get("framework", "Listen actively and validate feelings.")
            else:
                distortion_name = "General Support"
                definition = "No specific distortion detected."
                framework = "Listen empathetically, validate feelings, and respond with warmth."
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            distortion_name = "General Support"
            definition = "No specific distortion detected."
            framework = "Listen empathetically, validate feelings, and respond with warmth."

        # 2. CONSTRUCT dynamic system prompt
        system_instruction = (
            "You are Manas Mitra, a compassionate, empathetic, and supportive mental health companion. "
            "Your goal is to listen actively, validate the user's feelings, and respond with warmth, kindness, and understanding. "
            "Do not offer clinical diagnoses or prescribe medication. Keep your responses concise (normally 2-3 sentences).\n\n"
            "A cognitive distortion has been retrieved from the user's input to guide your response:\n"
            f"- Detected Distortion: {distortion_name}\n"
            f"- Clinical Definition: {definition}\n"
            f"- Therapeutic Framework: {framework}\n\n"
            "Apply this framework gently. If the user is greeting you, respond warmly without challenging a distortion."
        )

        # 3. CALL Gemini API (blocking call, offloaded to thread)
        return await asyncio.to_thread(_call_gemini, system_instruction, message)

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm sorry, I'm having trouble processing your request right now. Could you try again later?"

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        user_message = chat_request.message
        emotion = detect_emotion(user_message)
        reply = await generate_response(user_message, emotion)
        return {"emotion": emotion, "reply": reply}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Sorry, I'm having trouble processing your request. Please try again."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
