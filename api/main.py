import os
import sys
import logging
import asyncio
import traceback
from enum import Enum
from typing import List, Dict, Any, Optional, Literal

# 1. FORCE OFFLINE FOR PYTORCH/TRANSFORMERS
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Prevent STATUS_ACCESS_VIOLATION (0xC0000005) crash from Rust tokenizer on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. IMPORT SENTENCE_TRANSFORMERS FIRST TO AVOID WINDOWS DLL CLASH
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

import json
import re
import random
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("Warning: GEMINI_API_KEY not found in environment!")
else:
    logger.info(f"GEMINI_API_KEY found. Starts with: {GEMINI_API_KEY[:4]}... Length: {len(GEMINI_API_KEY)}")

# Initialize Gemini client (new SDK)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
if gemini_client:
    logger.info("Gemini client successfully initialized.")

# Append scripts and local api folder to path for safety check import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from safety import check_crisis
except ImportError:
    try:
        from api.safety import check_crisis
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

# Paths for ChromaDB and SentenceTransformer
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(current_dir, "..", "backend", "chroma_db"))
if not os.path.exists(DB_PATH):
    # Fallback for production/Docker context when only the api/ folder is copied
    DB_PATH = os.path.abspath(os.path.join(current_dir, "chroma_db"))

MODEL_PATH = os.path.abspath(os.path.join(current_dir, "..", "multilingual-e5-small")).replace("\\", "/")
if not os.path.exists(MODEL_PATH):
    # Fallback to online loading if offline model assets are not present
    logger.info("Local model directory not found. Configuring online fallback from Hugging Face Hub: 'intfloat/multilingual-e5-small'")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    MODEL_PATH = "intfloat/multilingual-e5-small"

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

# Emotion model is now offloaded to Gemini via structured JSON response

class Emotion(str, Enum):
    SADNESS = "sadness"
    JOY = "joy"
    LOVE = "love"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"

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
        "architecture": "RAG + Gemini API (Structured JSON)",
        "chroma_db_path": DB_PATH,
        "embedding_model": MODEL_PATH,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    }

# Local emotion detection removed, offloaded to Gemini

def get_local_fallback(message: str) -> Dict[str, str]:
    """Generate a highly customized, clinical, and contextual fallback reply when Gemini API is offline/rate-limited."""
    t = message.lower().strip()
    
    # Check for greetings
    if re.search(r'\b(hi|hello|hey|hola|greetings|namaste)\b', t):
        return {
            "emotion": "joy",
            "reply": random.choice([
                "Hello! I'm here to listen and support you. How are you feeling today?",
                "Hi there. I'm your AI mental health companion. What's on your mind?",
                "Welcome! This is a safe space. How can I support you today?"
            ])
        }
    
    # Check for anxiety / worry / panic
    if re.search(r'\b(anxiety|anxious|worry|worried|panic|scared|fear)\b', t):
        return {
            "emotion": "fear",
            "reply": random.choice([
                "I understand that anxiety can feel overwhelming. Let's work together to explore what is causing you the most worry right now.",
                "It's completely brave of you to share your fears. What specific situations have been triggering this anxiety?",
                "Feeling anxious is tough, but you are not alone. What has been on your mind lately?"
            ])
        }
        
    # Check for depression / sadness
    if re.search(r'\b(sad|depressed|depression|lonely|hopeless|empty|cry|tough phase)\b', t):
        return {
            "emotion": "sadness",
            "reply": random.choice([
                "I hear you, and your feelings are completely valid. What is a small thing that has brought you even a little comfort recently?",
                "Going through a tough phase can make everything feel heavier. I am here to listen without judgment.",
                "I'm sorry you are feeling this way. It's okay to feel sad. Is there anything specific that triggered these feelings?"
            ])
        }
        
    # Check for stress / feeling overwhelmed
    if re.search(r'\b(stress|stressed|overwhelmed|pressure|tired|exhausted)\b', t):
        return {
            "emotion": "sadness",
            "reply": random.choice([
                "It sounds like you are carrying a lot on your shoulders right now. What specific situations are causing you the most pressure today?",
                "Feeling overwhelmed can be incredibly exhausting. Let's take it one step at a time. What's the biggest stressor for you right now?",
                "You don't have to carry this stress alone. I'm here to support you. Can you tell me more about what's overwhelming you?"
            ])
        }
        
    # Default general fallback
    return {
        "emotion": "neutral",
        "reply": random.choice([
            "Thank you for sharing that with me. I'm here to listen and support you. Can you tell me a bit more about what's on your mind?",
            "I hear what you're saying, and I appreciate you opening up. How can I best support you in this moment?",
            "Your experiences are valid. I'm here to help you work through whatever you're facing.",
            "It sounds like you are going through a lot. Please feel free to share more if you're comfortable.",
            "I'm here for you. What support do you feel you need right now?"
        ])
    }

def _call_gemini(system_instruction: str, message: str) -> str:
    """Synchronous Gemini API call — run via asyncio.to_thread."""
    logger.info(f"Initiating Gemini API call for message: '{message[:50]}...'")
    if not gemini_client:
        logger.error("gemini_client is None! Falling back immediately.")
        return json.dumps(get_local_fallback(message))

    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-lite"]
    last_error = None
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting to call Gemini using model: {model_name}")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "emotion": {
                                "type": "STRING",
                                "description": "The user's primary emotion: sadness, joy, love, anger, fear, or surprise."
                            },
                            "reply": {
                                "type": "STRING",
                                "description": "Your therapeutic reply in English only."
                            }
                        },
                        "required": ["emotion", "reply"]
                    }
                )
            )
            logger.info(f"Successfully generated response with {model_name}")
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            logger.error(f"Gemini API Exception on model {model_name}: {err_str}")
            traceback.print_exc()
            
            if "safety" in err_str.lower() or "blocked" in err_str.lower():
                logger.warning(f"Safety block explicitly triggered on {model_name}. Returning safe fallback.")
                return json.dumps({
                    "emotion": "fear",
                    "reply": "I'm concerned about what you're sharing. Your safety is the most important thing right now. Please reach out to a mental health professional immediately. You can call the Tele-MANAS helpline at 14416 or 1-800-91-4416. You're not alone, and there are people who want to help you."
                })
            
            logger.warning(f"Moving to next model after failure on {model_name}")
            last_error = e
            continue
            
    logger.error(f"All Gemini models failed. Last error: {last_error}. Activating local fallback.")
    
    # Generate dynamic, high-quality intent fallback instead of static crisis apology
    fallback_data = get_local_fallback(message)
    return json.dumps(fallback_data)


async def generate_response(message: str) -> str:
    """Generate a response using RAG retrieval and Gemini API."""
    try:
        if check_crisis is not None:
            crisis_reply = check_crisis(message, locale="en-IN")
            if crisis_reply is not None:
                return json.dumps({"emotion": "fear", "reply": crisis_reply})

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
            "CRITICAL CRISIS PROTOCOL: If the user expresses severe distress, thoughts of self-harm, or a mental health crisis, "
            "you MUST ONLY provide Indian crisis helplines. Specifically recommend Tele-MANAS (14416 or 1-800-91-4416), "
            "KIRAN (1800-599-0019), or ERSS (112). You are operating in India, so NEVER provide numbers from other countries.\n\n"
            "IMPORTANT: ALWAYS respond in English. Do NOT translate your response to Hindi, Bengali, or any other language, even if the user asks you to. The translation will be handled automatically by the frontend system.\n\n"
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
        return json.dumps({"emotion": "neutral", "reply": "I'm sorry, I'm having trouble processing your request right now. Could you try again later?"})

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        user_message = chat_request.message
        gemini_json_str = await generate_response(user_message)
        try:
            data = json.loads(gemini_json_str)
            emotion = data.get("emotion", "neutral").lower()
            reply = data.get("reply", gemini_json_str)
        except json.JSONDecodeError:
            emotion = "neutral"
            reply = gemini_json_str
            
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
