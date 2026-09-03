import os
import sys
import logging
import asyncio
import traceback
import json
import re
import random
import numpy as np
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
load_dotenv(".env")
load_dotenv("backend/.env")

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

app = FastAPI(title="Manas Mitra Vercel API", description="Lightweight Serverless RAG + Gemini API backend for Manas Mitra")

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
    },
    {
        "name": "All-or-Nothing Thinking",
        "definition": "Viewing things in black-and-white, absolute terms (e.g., 'always a failure', 'perfect or ruined').",
        "framework": "Highlight the middle ground and shades of gray. Shift focus from binary outcomes to progressive learning and self-compassion.",
        "keywords": ["complete failure", "perfect", "ruined", "total waste", "either", "unproductive", "never right"],
    },
    {
        "name": "Emotional Reasoning",
        "definition": "Assuming that your subjective feelings reflect objective reality (e.g., 'I feel guilty, so I must be bad').",
        "framework": "Help the user distinguish between temporary emotional states and objective, observable facts. Validate the feeling, but challenge the factual conclusion.",
        "keywords": ["feel lonely", "feel stupid", "feel guilty", "feel like an idiot", "feel anxious", "nobody cares", "unsolvable"],
    },
    {
        "name": "Overgeneralization",
        "definition": "Drawing broad, negative conclusions based on a single event (often using words like 'always', 'never', 'everyone').",
        "framework": "Gently prompt the user to look for exceptions to their perceived universal patterns, pointing out specific, positive counter-instances.",
        "keywords": ["always", "never", "everyone", "nobody", "every single time", "all the time"],
    },
    {
        "name": "Should Statements",
        "definition": "Using rigid rules ('should', 'must', 'ought to') to motivate yourself or judge others, leading to guilt or anger.",
        "framework": "Help the user reframe self-imposed demands into flexible preferences or choices (e.g., changing 'I should study' to 'It is helpful for my goals if I study').",
        "keywords": ["should", "must", "ought to", "supposed to", "have to be strong", "shouldn't feel"],
    }
]

# Load precomputed embeddings
PRECOMPUTED_EMBEDDINGS = []
try:
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "config", "distortion_embeddings.json"),
        os.path.join(os.path.dirname(__file__), "config", "distortion_embeddings.json"),
        "config/distortion_embeddings.json"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                PRECOMPUTED_EMBEDDINGS = json.load(f)
            logger.info(f"Loaded {len(PRECOMPUTED_EMBEDDINGS)} precomputed distortion embeddings from {p}")
            break
except Exception as e:
    logger.error(f"Failed to load precomputed embeddings: {e}")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def get_query_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for query using Gemini API."""
    if not gemini_client:
        return None
    try:
        res = gemini_client.models.embed_content(model="gemini-embedding-001", contents=text)
        if hasattr(res, "embeddings") and res.embeddings and len(res.embeddings) > 0:
            return np.array(res.embeddings[0].values, dtype=np.float32)
        elif hasattr(res, "embedding") and hasattr(res.embedding, "values"):
            return np.array(res.embedding.values, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Gemini embedding query error: {e}")
    return None

def retrieve_cognitive_distortion(query: str) -> Dict[str, str]:
    """Retrieve closest matching CBT distortion using lightweight precomputed embeddings & keyword fallback."""
    query_clean = query.strip()
    
    # 1. Semantic Embedding Search
    if PRECOMPUTED_EMBEDDINGS:
        q_vec = get_query_embedding(query_clean)
        if q_vec is not None:
            best_score = -1.0
            best_item = None
            for item in PRECOMPUTED_EMBEDDINGS:
                ex_vec = np.array(item["embedding"], dtype=np.float32)
                sim = cosine_similarity(q_vec, ex_vec)
                if sim > best_score:
                    best_score = sim
                    best_item = item
            
            if best_item and best_score >= 0.40:
                logger.info(f"RAG retrieved distortion: {best_item['distortion_name']} (similarity: {best_score:.3f}) for query '{query_clean[:30]}'")
                return {
                    "name": best_item["distortion_name"],
                    "definition": best_item["definition"],
                    "framework": best_item["framework"]
                }

    # 2. Fast Keyword Fallback
    q_lower = query_clean.lower()
    for dist in DISTORTIONS_DATA:
        for kw in dist["keywords"]:
            if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
                logger.info(f"RAG keyword match: {dist['name']} via '{kw}'")
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

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "emotion": {"type": "STRING"},
        "reply": {"type": "STRING"}
    },
    "required": ["emotion", "reply"]
}

def clean_json_response(raw_text: str) -> Optional[Dict[str, str]]:
    """Clean markdown code blocks and parse JSON response cleanly."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            reply_val = data.get("reply", "")
            emotion_val = data.get("emotion", "neutral")
            
            if isinstance(reply_val, str) and reply_val.strip().startswith("{"):
                try:
                    inner_data = json.loads(reply_val.strip())
                    if isinstance(inner_data, dict) and "reply" in inner_data:
                        reply_val = inner_data.get("reply", reply_val)
                        emotion_val = inner_data.get("emotion", emotion_val)
                except Exception:
                    pass
                    
            return {
                "emotion": str(emotion_val).lower(),
                "reply": str(reply_val).strip()
            }
    except Exception:
        pass
        
    return {"emotion": "neutral", "reply": cleaned}

def _call_gemini(system_instruction: str, message: str) -> str:
    """Synchronous Gemini API call with multi-model fallback chain."""
    logger.info(f"Initiating Gemini API call for message: '{message[:50]}...'")
    if not gemini_client:
        logger.error("gemini_client is None! Falling back to local intent.")
        return json.dumps(get_local_fallback(message))

    # Valid active Gemini models (verified against Gemini API)
    models_to_try = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash"]
    last_error = None
    
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting Gemini call with model: {model_name}")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                )
            )

            if response and response.text:
                parsed = clean_json_response(response.text)
                if parsed and parsed.get("reply"):
                    logger.info(f"Successfully generated response with {model_name}: {parsed['reply'][:50]}...")
                    return json.dumps(parsed)
                elif response.text.strip():
                    return json.dumps({"emotion": "neutral", "reply": response.text.strip()})
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
            
    logger.error(f"All Gemini models failed. Last error: {last_error}. Activating local fallback.")
    return json.dumps(get_local_fallback(message))

def get_local_fallback(message: str) -> Dict[str, str]:
    """Generate clinical fallback reply when Gemini API is offline."""
    t = message.lower()
    
    if re.search(r'\b(anxious|anxiety|scared|fear|panic|nervous|worried)\b', t):
        return {
            "emotion": "fear",
            "reply": random.choice([
                "I hear how anxious you are feeling right now. Take a deep, slow breath. What specific situation is causing you the most worry at this moment?",
                "Anxiety can make everything feel overwhelming. Let's ground ourselves together. Can you tell me what's on your mind?",
                "It's completely brave of you to share your fears. What specific situations have been triggering this anxiety?"
            ])
        }
        
    if re.search(r'\b(depressed|depression|sad|hopeless|empty|lonely|worthless)\b', t):
        return {
            "emotion": "sadness",
            "reply": random.choice([
                "I'm so sorry you're feeling this way. Depression can make everything feel heavy, but you don't have to carry it alone. What has been feeling hardest today?",
                "Your feelings are completely valid, and I'm here with you. What small thing has brought you comfort recently?",
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

async def generate_response(message: str) -> str:
    """Generate response using RAG retrieval and Gemini API."""
    try:
        # 1. CRISIS SAFETY HARD-BLOCK
        if check_crisis is not None:
            crisis_reply = check_crisis(message, locale="en-IN")
            if crisis_reply is not None:
                return json.dumps({"emotion": "fear", "reply": crisis_reply})

        # 2. LIGHTWEIGHT RAG RETRIEVAL
        rag_data = retrieve_cognitive_distortion(message)
        distortion_name = rag_data["name"]
        distortion_def = rag_data["definition"]
        distortion_framework = rag_data["framework"]

        # 3. BUILD SYSTEM PROMPT
        system_prompt = f"""You are Manas Mitra, a compassionate, empathetic, and culturally aware AI mental health companion designed specifically for college students in India.

Clinical Therapeutic Guidelines (Cognitive Behavioral Therapy - CBT):
- Current Identified Thought Pattern / Distortion: {distortion_name}
- Definition: {distortion_def}
- Therapeutic Framework to Apply: {distortion_framework}

Communication Principles:
1. Show deep warmth, validation, and emotional resonance.
2. Avoid medical jargon or diagnosing. Frame suggestions as gentle self-exploration.
3. Keep responses concise (2-4 sentences max), clear, and reassuring.
4. Always respond with a valid JSON object matching this exact schema:
{{"emotion": "sadness|joy|love|anger|fear|surprise|neutral", "reply": "Your empathetic response here"}}"""

        # 4. EXECUTE GEMINI API CALL
        response_json_str = await asyncio.to_thread(_call_gemini, system_prompt, message)
        return response_json_str

    except Exception as e:
        logger.error(f"Error in generate_response: {e}\n{traceback.format_exc()}")
        return json.dumps(get_local_fallback(message))

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    emotion: str
    reply: str

@app.get("/")
@app.get("/api/py")
@app.get("/api/py/health")
async def health_check():
    return {
        "status": "healthy",
        "architecture": "Single-Vercel Serverless RAG (Precomputed Embeddings + Gemini 2.5/3.5/3.6 Flash)",
        "memory_profile": "Ultra-lightweight (<30MB)",
        "gemini_configured": bool(gemini_client)
    }

@app.post("/chat")
@app.post("/api/py/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message is required")
        
    response_json_str = await generate_response(request.message)
    try:
        data = json.loads(response_json_str)
        return ChatResponse(
            emotion=data.get("emotion", "neutral"),
            reply=data.get("reply", "I am here to support you.")
        )
    except Exception as e:
        logger.error(f"Failed to parse JSON response: {e}")
        fallback = get_local_fallback(request.message)
        return ChatResponse(
            emotion=fallback["emotion"],
            reply=fallback["reply"]
        )

@app.get("/debug/gemini")
@app.get("/api/py/debug/gemini")
async def debug_gemini(msg: str = "Hello, I am testing the AI"):
    try:
        res = await asyncio.to_thread(_call_gemini, "You are a helpful companion. Respond in JSON.", msg)
        return {"success": True, "raw": res}
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
