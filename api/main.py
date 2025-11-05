from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import os

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
BASE_MODEL = "google/flan-t5-base"  # Base model name
MODEL_PATH = "../outputs/lora-manas-mitra-irec/checkpoint-66"  # Path to your fine-tuned model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and base model
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load the base model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Load the LoRA weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model = model.merge_and_unload()  # Merge LoRA weights with base model
    
    model.eval()
    logger.info(f"Model loaded on {DEVICE} with LoRA weights from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: ChatMessage

@app.get("/")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME, "device": DEVICE}

def format_conversation(messages: List[ChatMessage]) -> str:
    """Format the conversation history for the model."""
    # Only include the last 3 messages to avoid context window issues
    recent_messages = messages[-3:]
    
    # Format as a conversation
    formatted = []
    for msg in recent_messages:
        if msg.role == 'user':
            formatted.append(f"User: {msg.content}")
        else:
            formatted.append(f"Assistant: {msg.content}")
    
    # Add a prompt for the next response
    formatted.append("Assistant:")
    return "\n".join(formatted)

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        # Format the conversation
        conversation = format_conversation(chat_request.messages)
        
        # Tokenize the input
        inputs = tokenizer(
            conversation,
            return_tensors="pt",
            max_length=384,
            truncation=True,
            padding=True,
            add_special_tokens=True
        ).to(DEVICE)
        
        # Generate response with adjusted parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                num_beams=3,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and clean the response
        response_text = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Sometimes the model repeats the conversation, so we'll clean that up
        if "Assistant:" in response_text:
            response_text = response_text.split("Assistant:")[-1].strip()
        
        # Remove any remaining role prefixes
        for prefix in ["User:", "Assistant:"]:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
        
        return {
            "message": {
                "role": "assistant",
                "content": response_text
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
