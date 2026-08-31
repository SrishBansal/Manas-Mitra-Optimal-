import os
import sys
import logging
import asyncio
import traceback
import json
import re
import random
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
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

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
if gemini_client:
    logger.info("Gemini client successfully initialized.")

# Import safety check
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from safety import check_crisis
except ImportError:
    try:
        from api.safety import check_crisis
    except ImportError:
        check_crisis = None

app = FastAPI(title="Manas Mitra API", description="Lightweight RAG + Gemini API backend for the Manas Mitra mental health chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# CLINICAL COGNITIVE DISTORTIONS KNOWLEDGE BASE (CBT RAG)
# -------------------------------------------------------------
DISTORTIONS_DATA = [
    {
        "name": "Catastrophizing",
        "definition": "Expecting the absolute worst outcome in a situation, even with minimal evidence.",
        "framework": "Gently explore the actual probability of the feared outcome. Encourage the user to consider the best-case, worst-case, and most likely scenarios.",
        "keywords": ["fail", "ruined", "disaster", "heart attack", "die", "worst", "hate me", "fired", "end of the world", "hopeless"],
        "examples": [
            "I'm going to fail this exam and my life will be completely ruined.",
            "If I make one mistake in my presentation, I'll be fired and end up homeless.",
            "Everything is going wrong, this is a total disaster and there's no way out.",
            "They haven't texted back, they must hate me and our friendship is over.",
            "My chest feels tight, I'm probably having a heart attack and going to die."
        ]
    },
    {
        "name": "All-or-Nothing Thinking",
        "definition": "Viewing things in black-and-white, absolute terms (e.g., 'always a failure', 'perfect or ruined').",
        "framework": "Highlight the middle ground and shades of gray. Shift focus from binary outcomes to progressive learning and self-compassion.",
        "keywords": ["complete failure", "perfect", "ruined", "total waste", "either", "unproductive", "never right"],
        "examples": [
            "If I don't get an A+, I'm a complete failure.",
            "I ate one cookie, so my entire diet is ruined now.",
            "Either they completely agree with me or they hate my guts.",
            "I messed up one slide, the whole presentation was a disaster.",
            "I couldn't finish all my tasks today, so I was totally unproductive."
        ]
    },
    {
        "name": "Emotional Reasoning",
        "definition": "Assuming that your subjective feelings reflect objective reality (e.g., 'I feel guilty, so I must be bad').",
        "framework": "Help the user distinguish between temporary emotional states and objective, observable facts. Validate the feeling, but challenge the factual conclusion.",
        "keywords": ["feel lonely", "feel stupid", "feel guilty", "feel like an idiot", "feel anxious", "nobody cares", "unsolvable"],
        "examples": [
            "I feel so lonely, so nobody must care about me.",
            "I feel like an idiot, which means I am stupid.",
            "I'm feeling so anxious, so something terrible is definitely about to happen.",
            "I feel guilty, so I must have done something awful to them.",
            "I feel overwhelmed, so this problem must be completely unsolvable."
        ]
    },
    {
        "name": "Overgeneralization",
        "definition": "Drawing broad, negative conclusions based on a single event (often using words like 'always', 'never', 'everyone').",
        "framework": "Gently prompt the user to look for exceptions to their perceived universal patterns, pointing out specific, positive counter-instances.",
        "keywords": ["always", "never", "everyone", "nobody", "every single time", "all the time"],
        "examples": [
            "I always mess up my relationships, I'll be alone forever.",
            "Nothing ever goes right for me in this city.",
            "I failed this job interview, so I will never get hired anywhere.",
            "Everyone is always happier and more successful than me.",
            "I can never do anything right."
        ]
    },
    {
        "name": "Should Statements",
        "definition": "Using rigid rules ('should', 'must', 'ought to') to motivate yourself or judge others, leading to guilt or anger.",
        "framework": "Help the user reframe self-imposed demands into flexible preferences or choices (e.g., changing 'I should study' to 'It is helpful for my goals if I study').",
        "keywords": ["should", "must", "ought to", "supposed to", "have to be strong", "shouldn't feel"],
        "examples": [
            "I should be studying right now instead of resting, I'm so lazy.",
            "I must always be strong and never show any weakness.",
            "I shouldn't feel sad about this minor thing, I should be happier.",
            "They should know what's wrong without me telling them.",
            "I ought to have figured my life out by now."
        ]
    }
]

# Vector Store Cache for RAG
cached_embeddings: Dict[str, np.ndarray] = {}

def get_embedding(text: str) -> Optional[np.ndarray]:
    """Compute text embedding using Gemini API (0 MB local RAM footprint)."""
    if not gemini_client:
        return None
    try:
        response = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        if response.embedding and response.embedding.values:
            return np.array(response.embedding.values, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Gemini embedding error: {e}")
    return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def retrieve_cognitive_distortion(query: str) -> Dict[str, str]:
    """Retrieve the closest matching clinical CBT distortion via semantic vector RAG with keyword fallback."""
    query_clean = query.strip()
    
    # 1. Semantic Embedding Retrieval via Gemini Embeddings
    query_vec = get_embedding(query_clean)
    if query_vec is not None:
        best_score = -1.0
        best_distortion = None
        
        for dist in DISTORTIONS_DATA:
            for example in dist["examples"]:
                if example not in cached_embeddings:
                    emb = get_embedding(example)
                    if emb is not None:
                        cached_embeddings[example] = emb
                
                ex_vec = cached_embeddings.get(example)
                if ex_vec is not None:
                    sim = cosine_similarity(query_vec, ex_vec)
                    if sim > best_score:
                        best_score = sim
                        best_distortion = dist
                        
        if best_distortion and best_score >= 0.55:
            logger.info(f"RAG retrieved distortion: {best_distortion['name']} (similarity: {best_score:.3f})")
            return {
                "name": best_distortion["name"],
                "definition": best_distortion["definition"],
                "framework": best_distortion["framework"]
            }

    # 2. Fast Keyword / Intent matching fallback
    q_lower = query_clean.lower()
    for dist in DISTORTIONS_DATA:
        for kw in dist["keywords"]:
            if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
                logger.info(f"RAG keyword match: {dist['name']} via keyword '{kw}'")
                return {
                    "name": dist["name"],
                    "definition": dist["definition"],
                    "framework": dist["framework"]
                }

    # 3. Default General Support
    return {
        "name": "General Support",
        "definition": "No specific cognitive distortion detected.",
        "framework": "Listen empathetically, validate the user's emotional state, and respond with warmth."
    }

# -------------------------------------------------------------
# MODELS & SCHEMAS
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    emotion: str
    reply: str

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "architecture": "Lightweight RAG (Gemini Embeddings + Gemini 2.0 Flash)",
        "memory_profile": "Ultra-lightweight (<60MB)",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    }

def get_local_fallback(message: str) -> Dict[str, str]:
    """Generate high-quality clinical fallback reply when Gemini API is offline."""
    t = message.lower().strip()
    
    if re.search(r'\b(hi|hello|hey|hola|greetings|namaste)\b', t):
        return {
            "emotion": "joy",
            "reply": random.choice([
                "Hello! I'm here to listen and support you. How are you feeling today?",
                "Hi there. I'm your AI mental health companion. What's on your mind?",
                "Welcome! This is a safe space. How can I support you today?"
            ])
        }
    
    if re.search(r'\b(anxiety|anxious|worry|worried|panic|scared|fear)\b', t):
        return {
            "emotion": "fear",
            "reply": random.choice([
                "I understand that anxiety can feel overwhelming. Let's work together to explore what is causing you the most worry right now.",
                "It's completely brave of you to share your fears. What specific situations have been triggering this anxiety?",
                "Feeling anxious is tough, but you are not alone. What has been on your mind lately?"
            ])
        }
        
    if re.search(r'\b(sad|depressed|depression|lonely|hopeless|empty|cry|tough phase)\b', t):
        return {
            "emotion": "sadness",
            "reply": random.choice([
                "I hear you, and your feelings are completely valid. What is a small thing that has brought you even a little comfort recently?",
                "Going through a tough phase can make everything feel heavier. I am here to listen without judgment.",
                "I'm sorry you are feeling this way. It's okay to feel sad. Is there anything specific that triggered these feelings?"
            ])
        }
        
    if re.search(r'\b(stress|stressed|overwhelmed|pressure|tired|exhausted)\b', t):
        return {
            "emotion": "sadness",
            "reply": random.choice([
                "It sounds like you are carrying a lot on your shoulders right now. What specific situations are causing you the most pressure today?",
                "Feeling overwhelmed can be incredibly exhausting. Let's take it one step at a time. What's the biggest stressor for you right now?",
                "You don't have to carry this stress alone. I'm here to support you. Can you tell me more about what's overwhelming you?"
            ])
        }
        
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
    """Synchronous Gemini API call with multi-model fallback pipeline."""
    logger.info(f"Initiating Gemini API call for message: '{message[:50]}...'")
    if not gemini_client:
        logger.error("gemini_client is None! Falling back immediately.")
        return json.dumps(get_local_fallback(message))

    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-lite"]
    last_error = None
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting Gemini call with: {model_name}")
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
            logger.error(f"Gemini API error on {model_name}: {err_str}")
            if "safety" in err_str.lower() or "blocked" in err_str.lower():
                return json.dumps({
                    "emotion": "fear",
                    "reply": "I'm concerned about what you're sharing. Your safety is the most important thing right now. Please reach out to a mental health professional immediately. You can call the Tele-MANAS helpline at 14416 or 1-800-91-4416. You're not alone, and there are people who want to help you."
                })
            last_error = e
            continue
            
    logger.error(f"All Gemini models failed. Activating local fallback.")
    return json.dumps(get_local_fallback(message))

async def generate_response(message: str) -> str:
    """Generate response using RAG retrieval and Gemini API."""
    try:
        # 1. CRISIS SAFETY HARD-BLOCK
        if check_crisis is not None:
            crisis_reply = check_crisis(message, locale="en-IN")
            if crisis_reply is not None:
                return json.dumps({"emotion": "fear", "reply": crisis_reply})

        # 2. RAG RETRIEVAL (CBT Cognitive Distortion Match)
        rag_data = retrieve_cognitive_distortion(message)
        distortion_name = rag_data["name"]
        definition = rag_data["definition"]
        framework = rag_data["framework"]

        # 3. DYNAMIC THERAPEUTIC SYSTEM PROMPT
        system_instruction = (
            "You are Manas Mitra, a compassionate, empathetic, and supportive mental health companion for college students in India. "
            "Your goal is to listen actively, validate the user's feelings, and respond with warmth, kindness, and understanding. "
            "Do not offer clinical diagnoses or prescribe medication. Keep your responses concise (normally 2-3 sentences).\n\n"
            "CRITICAL CRISIS PROTOCOL: If the user expresses severe distress, thoughts of self-harm, or a mental health crisis, "
            "you MUST ONLY provide Indian crisis helplines: Tele-MANAS (14416 or 1-800-91-4416), KIRAN (1800-599-0019), or ERSS (112).\n\n"
            "IMPORTANT: ALWAYS respond in English. Do NOT translate your response to Hindi or other languages; translation is handled by the frontend.\n\n"
            "A cognitive distortion has been retrieved from the user's input to guide your therapeutic response:\n"
            f"- Detected Distortion: {distortion_name}\n"
            f"- Clinical Definition: {definition}\n"
            f"- Therapeutic Framework: {framework}\n\n"
            "Apply this framework gently. If the user is greeting you, respond warmly without challenging a distortion."
        )

        # 4. CALL GEMINI
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
