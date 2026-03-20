from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import os
from enum import Enum

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
    
    logger.info("Loading response generation model...")
    response_tokenizer = AutoTokenizer.from_pretrained(RESPONSE_MODEL)
    response_model = AutoModelForSeq2SeqLM.from_pretrained(RESPONSE_MODEL).to(DEVICE)
    response_model.eval()
    
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
        prompt = f"""You are a mental health assistant. The user is feeling {emotion}.
        
        User: {message}
        Assistant:"""
        
        # Tokenize and generate response
        inputs = response_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = response_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_beams=3,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode and clean the response
        response = response_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Remove the prompt from the response if it's included
        if prompt in response:
            response = response.replace(prompt, "")
        
        return response.strip()
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
