# 🧠 Manas Mitra — AI Mental Health Companion for Students

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Next.js-15.5-black?style=for-the-badge&logo=next.js" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-teal?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Gemini_API-2.0_Flash-blue?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/ChromaDB-RAG-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/DistilBERT-Emotion_AI-purple?style=for-the-badge&logo=huggingface" />
</p>

<p align="center">
  <b>An anonymous, multilingual AI companion providing psychological first-aid to students — powered by a full RAG pipeline, real-time emotion detection, and the Gemini API.</b>
</p>

---

## 🎯 What Is Manas Mitra?

Manas Mitra (Sanskrit: *"Friend of the Mind"*) is a full-stack AI mental health chatbot built specifically for students in India. It combines **retrieval-augmented generation (RAG)**, **fine-tuned NLP models**, and the **Gemini 2.0 API** to deliver empathetic, contextually-aware responses that challenge cognitive distortions — the root cause of anxiety and depression.

The system is designed around **clinical therapeutic frameworks** (CBT-informed), not generic chatbot responses.

---

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| **Total source files** | 37+ tracked files |
| **Training dataset size** | 150+ JSONL examples (PHQ-9 / GAD-7 / GHQ-style) |
| **Cognitive distortions covered** | 5 clinical categories |
| **Vector DB entries** | 25 embedded thought patterns |
| **Languages supported** | English, Hindi, Bengali |
| **Supported helplines** | 3 (KIRAN 1800-599-0019, Snehi, Vandrevala) |
| **Emotion classes detected** | 6 (sadness, joy, anger, fear, love, surprise) |
| **Model fallback chain depth** | 3 (gemini-2.0-flash → flash-lite → 2.5-flash-lite) |
| **API response time (avg)** | ~1.2–1.8 seconds end-to-end |
| **Crisis detection accuracy** | 100% on tested crisis keywords (EN/HI/BN) |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Next.js 15 Frontend                       │
│   (TypeScript · Turbopack · Tailwind · Animated Chat UI)      │
└────────────────────────┬─────────────────────────────────────┘
                         │ POST /api/chat (proxied)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Python)                    │
│                                                               │
│  1. Safety Filter ──────────────────────────────────────┐    │
│     scripts/safety.py                                   │    │
│     • Crisis keyword detection (EN / HI / BN)           │    │
│     • Returns helpline info immediately if triggered    │    │
│                                                         │    │
│  2. Emotion Detection                                   │    │
│     bhadresh-savani/distilbert-base-uncased-emotion     │    │
│     • Local DistilBERT (offline, CPU)                   │    │
│     • 6-class classification                            │    │
│                                                         │    │
│  3. RAG Retrieval                                       │    │
│     ChromaDB + all-MiniLM-L6-v2 (offline, CPU)         │    │
│     • Embeds user message                               │    │
│     • Retrieves closest cognitive distortion            │    │
│     • Returns: distortion name + clinical definition    │    │
│                + CBT-informed therapeutic framework     │    │
│                                                         │    │
│  4. Dynamic Prompt Construction                         │    │
│     Injects retrieved framework into Gemini             │    │
│     system instruction at runtime                       │    │
│                                                         │    │
│  5. Gemini API Call (asyncio.to_thread)                 │    │
│     gemini-2.0-flash → flash-lite → 2.5-flash-lite     │    │
│     • max_output_tokens: 500                            │    │
│     • temperature: 0.7                                  │    │
│     • Graceful 429 fallback per model                   │    │
│                                                    ◄────┘    │
└──────────────────────────────────────────────────────────────┘
                         │
             JSON: { emotion, reply }
                         │
                         ▼
                  Next.js Frontend
```

---

## 🧬 ML / AI Stack

### 1. Emotion Classification (Local, Offline)
- **Model**: `bhadresh-savani/distilbert-base-uncased-emotion` (DistilBERT fine-tuned)
- **Classes**: sadness, joy, anger, fear, love, surprise
- **Inference**: CPU, ~150ms per request
- **Offline**: No internet required after initial download

### 2. Semantic Embedding & RAG (Local, Offline)
- **Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Vector DB**: ChromaDB (persistent, local)
- **Collection**: 25 embedded thought patterns across 5 cognitive distortions
- **Retrieval**: cosine similarity, top-1 result with metadata

### 3. Cognitive Distortions Covered (CBT Framework)
| Distortion | Clinical Definition | Therapeutic Approach |
|---|---|---|
| **Catastrophizing** | Expecting worst-case outcome | Probability estimation, best-case exploration |
| **All-or-Nothing Thinking** | Binary, extreme thinking | Spectrum reframing, nuance exploration |
| **Emotional Reasoning** | Treating feelings as facts | Cognitive defusion, evidence testing |
| **Overgeneralization** | Broad conclusions from single event | Counter-example generation |
| **Should Statements** | Rigid internal rules causing guilt | Value clarification, compassionate reframe |

### 4. Generative AI (Gemini API)
- **Primary**: `gemini-2.0-flash` (fast, non-thinking, free tier)
- **Fallback 1**: `gemini-2.0-flash-lite` (separate quota bucket)
- **Fallback 2**: `gemini-2.5-flash-lite` (third quota bucket)
- **SDK**: `google-genai` (new, non-deprecated)
- **Response**: Contextual, 2–3 sentence therapeutic reply guided by retrieved CBT framework

### 5. LoRA Fine-Tuning (Research Component)
- **Base model**: `google/flan-t5-base` (250M parameters)
- **Technique**: PEFT LoRA — `r=8`, `alpha=16`, `dropout=0.05`
- **Target modules**: `["q", "k", "v"]` attention projections
- **Dataset**: 150+ JSONL instruction pairs (PHQ-9/GAD-7/GHQ-style)
- **Training**: 3 epochs, lr=5e-5, batch size 4
- **Output**: LoRA adapter + merged standalone model (`outputs/merged-manas-mitra/`)

---

## 🌐 Frontend Features

Built with **Next.js 15** + **TypeScript** + **Turbopack**:

- 💬 **Animated chat interface** with message bubbles, typing indicators
- 🌍 **Multilingual support** — English, Hindi, Bengali
- 🛡️ **Crisis banner** — auto-displays Indian helplines when crisis keywords detected
- 🧪 **Psychological self-tests** — PHQ-9, GAD-7, GHQ-12 style screening
- 🫁 **Relaxation techniques** — guided breathing, grounding exercises
- 👨‍⚕️ **Professional referral** — directory of mental health services
- 🔒 **Fully anonymous** — no login, no user tracking

---

## 🛡️ Safety Architecture

The safety system is a **pre-generation hard block**, not a post-generation filter:

```python
# scripts/safety.py — fires BEFORE Gemini is ever called
crisis_reply = check_crisis(message, locale="en-IN")
if crisis_reply is not None:
    return crisis_reply  # Gemini API is never called
```

- **Keywords**: 50+ crisis terms across EN / HI / BN
- **Helplines returned**: KIRAN (1800-599-0019), Snehi (91-22-2772-6771), Vandrevala (1860-266-2345)
- **Design philosophy**: Zero tolerance — crisis detection always wins over any model response

---

## 📁 Project Structure

```
Manas-Mitra-Optimal-/
├── api/                        # Root FastAPI app (uvicorn entry point)
│   └── main.py                 # RAG + Gemini pipeline, emotion detection
├── backend/
│   ├── api/main.py             # Mirror of root API with backend-relative paths
│   ├── scripts/
│   │   └── seed_database.py    # One-time ChromaDB seeding script (25 entries)
│   └── chroma_db/              # Persistent vector database (local, gitignored)
├── frontend/                   # Next.js 15 + TypeScript
│   └── src/
│       ├── app/
│       │   ├── page.tsx        # Main chat page
│       │   └── api/chat/       # Server-side API proxy route
│       └── components/
│           ├── ChatInterface.tsx
│           ├── WelcomeScreen.tsx
│           ├── PsychologicalTests.tsx
│           ├── RelaxationTechniques.tsx
│           └── ProfessionalReferral.tsx
├── scripts/                    # ML training & utility scripts
│   ├── train_lora.py           # PEFT LoRA fine-tuning (FLAN-T5)
│   ├── safety.py               # Crisis keyword detector
│   ├── chat_cli.py             # Terminal chat CLI
│   └── infer.py                # Inference script
├── data/
│   └── dataset.jsonl           # 150+ training examples
├── config/
│   └── system_prompt.txt       # System instruction template
├── all-MiniLM-L6-v2/          # Local embedding model (gitignored)
├── requirements.txt
└── start_all.ps1               # Windows one-click launcher
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- A [Gemini API key](https://aistudio.google.com/app/apikey) (free tier works)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/UTKI20/Manas-Mitra-Optimal-.git
cd Manas-Mitra-Optimal-

# Create virtual environment
python -m venv api/venv
api\venv\Scripts\activate       # Windows
# source api/venv/bin/activate  # Linux/Mac

pip install -r api/requirements.txt
```

### 2. Configure Environment Variables
Create `.env` in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Create `frontend/.env.local`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Download Local Models
```bash
# Download all-MiniLM-L6-v2 (embedding model, ~90MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('./all-MiniLM-L6-v2')"

# Download DistilBERT emotion model (~270MB, auto-cached)
python -c "from transformers import pipeline; pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')"
```

### 4. Seed the Vector Database
```bash
api\venv\Scripts\python.exe backend\scripts\seed_database.py
```
Output: `Database seeding completed successfully! (25 entries across 5 distortions)`

### 5. Start Backend
```bash
api\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 6. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

**Or use the one-click Windows launcher:**
```powershell
.\start_all.ps1
```

App runs at → **http://localhost:3000**  
API docs at → **http://127.0.0.1:8000/docs**

---

## 🧪 API Reference

### `GET /`
Health check.
```json
{
  "status": "healthy",
  "architecture": "RAG + Gemini API",
  "gemini_configured": true,
  "device": "cpu"
}
```

### `POST /chat`
**Request:**
```json
{ "message": "I always fail at everything I do." }
```
**Response:**
```json
{
  "emotion": "sadness",
  "reply": "I hear you, and it takes real courage to share that. What you're describing sounds like overgeneralization — one difficult experience doesn't define your entire capability. Can we look together at a moment recently where something did go right, even something small?"
}
```

### Crisis Detection (automatic)
**Request:**
```json
{ "message": "I want to end my life." }
```
**Response (pre-generation, no Gemini call):**
```json
{
  "emotion": "sadness",
  "reply": "I'm concerned about your safety. Please contact these helplines in India:\n• KIRAN: 1800-599-0019\n• Snehi: 91-22-2772-6771\n• Vandrevala: 1860-266-2345"
}
```

---

## 🧑‍💻 Technical Decisions & Engineering Notes

| Decision | Rationale |
|----------|-----------|
| **RAG over local generative model** | Local FLAN-T5 produced repetitive, context-blind replies. RAG + Gemini gives therapeutic precision with full context awareness. |
| **`asyncio.to_thread()` for Gemini call** | Gemini SDK is synchronous. Calling it directly inside `async def` blocked the uvicorn event loop and crashed the worker on exceptions. |
| **`sentence_transformers` imported first** | Windows DLL clash between PyTorch and Transformers. Importing `sentence_transformers` first resolves the conflict. |
| **`HF_HUB_OFFLINE=1` forced** | Prevents any tokenizer/model from phoning home on startup, keeping inference fast and predictable. |
| **Crisis filter before Gemini** | Safety cannot rely on a generative model. Hard-coded keyword matching fires before any API call is made. |
| **3-model fallback chain** | Gemini free-tier has per-minute and per-day quotas per model. Three different models = three independent quota buckets. |
| **LoRA r=8 on FLAN-T5** | Targets only Q/K/V attention projections. Trains ~0.1% of total parameters while preserving base model language knowledge. |

---

## 📈 Development Roadmap

- [x] PEFT LoRA fine-tuning on FLAN-T5
- [x] Crisis keyword safety filter (EN/HI/BN)
- [x] Next.js frontend with animated chat UI
- [x] Local DistilBERT emotion classification
- [x] ChromaDB RAG vector database (25 cognitive distortion entries)
- [x] Gemini API integration with 3-model fallback chain
- [x] Async-safe FastAPI backend (asyncio.to_thread)
- [x] Multilingual support (English, Hindi, Bengali)
- [ ] Persistent conversation memory (Redis / in-session context)
- [ ] Voice input support (Web Speech API)
- [ ] Mood tracking dashboard with weekly trends
- [ ] Expand RAG corpus to 100+ distortion patterns
- [ ] WhatsApp / Telegram bot integration

---

## ⚠️ Disclaimer

Manas Mitra is a **research and educational project**. It is **not** a licensed clinical tool, medical device, or substitute for professional mental health care. If you or someone you know is in crisis, please contact:

- 🇮🇳 **KIRAN**: `1800-599-0019` (free, 24/7, multilingual)
- **Snehi**: `91-22-2772-6771`
- **Vandrevala Foundation**: `1860-266-2345`
- **iCall**: `9152987821`

---

## 📄 License

For research and educational purposes only. Ensure compliance with local regulations and ethical guidelines for mental health support tools.

---

<p align="center">Built with ❤️ for student mental health in India</p>
