from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import os
import google.generativeai as genai
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables.")



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Manas Mitra API", description="API for the Manas Mitra mental health chatbot")

# CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
EMOTION_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
RESPONSE_MODEL = "google/flan-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
try:
    logger.info("Loading emotion classification model...")
    emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL).to(DEVICE)
    emotion_model.eval()
    
    # RESPONSE_MODEL is no longer needed as we use Gemini
    # logger.info("Loading response generation model...")
    # response_tokenizer = AutoTokenizer.from_pretrained(RESPONSE_MODEL)
    # response_model = AutoModelForSeq2SeqLM.from_pretrained(RESPONSE_MODEL).to(DEVICE)
    # response_model.eval()
    
    logger.info(f"Models loaded on {DEVICE}")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

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
        "models": {
            "emotion": EMOTION_MODEL,
            "response": RESPONSE_MODEL
        },
        "device": DEVICE
    }

def detect_emotion(text: str) -> str:
    """Detect the emotion in the given text."""
    try:
        inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        
        # Get the predicted emotion
        predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
        emotion = emotion_model.config.id2label[predicted_class_idx]
        
        return emotion.lower()
    except Exception as e:
        logger.error(f"Error in emotion detection: {str(e)}")
        return "neutral"

def generate_response(message: str, emotion: str) -> str:
    """Generate a response based on the message and detected emotion."""
    try:
        # Create a prompt that includes the emotion context
        prompt = f"""You are a compassionate mental health assistant. The user is feeling {emotion}.
        
        User: {message}
        
        Provide a supportive, empathetic, and helpful response. Keep it concise (under 150 words) but meaningful.
        """
        
        if not GENAI_API_KEY:
            return "Server configuration error: Gemini API Key missing."

        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        
        return response.text.strip()

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I'm having trouble processing your request right now. Could you try again later?"

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        # Get the latest user message
        user_message = chat_request.message
        
        # Detect emotion from the user's message
        emotion = detect_emotion(user_message)
        
        # Generate a response based on the message and detected emotion
        reply = generate_response(user_message, emotion)
        
        return {
            "emotion": emotion,
            "reply": reply
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Sorry, I'm having trouble processing your request. Please try again."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
