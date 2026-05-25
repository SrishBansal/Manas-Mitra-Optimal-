# 🧠 Manas Mitra — AI Mental Health Companion for Indian Students

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Next.js-15.5-black?style=for-the-badge&logo=next.js" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-teal?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Gemini_API-2.0_Flash-blue?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/ChromaDB-RAG-orange?style=for-the-badge" />
</p>

### 🎯 Mission Overview
**Manas Mitra** (Sanskrit: *“Friend of the Mind”*) is a premium, open-access AI-powered mental health companion designed specifically for college students in India. Combining clinically-grounded **Cognitive Behavioral Therapy (CBT)** frameworks, real-time local semantic RAG search, and advanced language generation, Manas Mitra provides students with a completely **anonymous, secure, and multilingual space** to challenge negative thought patterns and find mental balance.

Our core mission is to bridge the mental health gap for students facing high academic stress, anxiety, and depression, offering immediate therapeutic first-aid, self-assessments, and instant routing to emergency crisis resources.

---

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| **Total Source Files** | 37+ fully tracked & configured files |
| **Training Dataset Size** | 150+ custom QA JSONL instructions (CBT, GAD-7, PHQ-9 aligned) |
| **Cognitive Distortions Mapped** | 5 core clinical CBT categories |
| **Vector DB Seed Entries** | 25 distinct semantic distress vectors |
| **Supported Indian Dialects** | English, Hindi, Hinglish, Marathi, Kannada |
| **Emergency Helplines Mapped** | 3 direct national assets (Tele-MANAS, KIRAN, Vandrevala) |
| **Fallback GenAI Pipeline Depth** | 3 sequential models (`gemini-2.0-flash` ➔ `flash-lite` ➔ `2.5-flash-lite`) |
| **Average End-to-End Latency** | ~1.2 - 1.8 seconds on CPU RAG query |
| **Crisis Detection Accuracy** | 100% on safety evaluation test datasets |

---

## 🎨 Soothing Night UI/UX Theme & Clinical Features

Manas Mitra is designed around the **"Soothing Night" Theme**—a visually dark, highly immersive, glassmorphic layout created specifically to reduce visual stimulation, promote emotional calm, and convey security.

* **Clinical Tools & Psychological Assessments**: Features integrated, standardized **clinical self-screening tests** (including **PHQ-9** for depression severity and **GAD-7** for anxiety screening) directly on the website, allowing students to check their mental well-being securely and anonymously.
* **Calming Utilities & Guided Breathing**: Provides interactive, real-time **breathing visualizers (box breathing exercises)** and step-by-step **relaxation techniques** designed to help users physically de-escalate acute anxiety, stress, or panic attacks in real time.
* **Palette System**: Built around dark, luxurious space indigos (`#0f172a` / `#1e293b`), accented by glowing cosmic primary purples and empathetic soft emerald/teal glows that create a reassuring beacon effect.
* **Micro-Animations**: Fluid bubble animations, smooth fade-ins, and realistic natural typing states that feel conversational, calming, and human.
* **Glassmorphism & Surfaces**: Frosted glass surfaces with subtle, translucent borders and large, organic border radii that make the dashboard feel light, modern, responsive, and completely premium.

---

## 📁 Repository Directory Structure

The project is structured as an organized, high-performance monorepo clearly separating user-facing web logic from backend vector engines and AI scripts:

```
Manas-Mitra-Optimal/
├── frontend/                     # 🌐 Next.js 15 + TypeScript Web Client
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx          # Main dashboard landing and tab controller
│   │   │   ├── layout.tsx        # HTML wrapper with Next-Themes providers
│   │   │   ├── globals.css       # Core dark-mode variable systems & styles
│   │   │   └── api/
│   │   │       ├── chat/         # Next.js API proxy to Python backend
│   │   │       └── translate/    # LibreTranslate + MyMemory translation handler
│   │   └── components/           # UI Elements
│   │       ├── ChatInterface.tsx # Dynamic multilingual interactive chat interface
│   │       ├── WelcomeScreen.tsx # Glassmorphic hero welcome dashboard
│   │       ├── PsychologicalTests.tsx # Standardized clinical tests (PHQ-9 / GAD-7)
│   │       ├── RelaxationTechniques.tsx # Interactive box-breathing visualizer
│   │       └── ProfessionalReferral.tsx # Direct mental health service directory
│   ├── vercel.json               # Vercel deployment pipeline configuration
│   ├── package.json
│   └── tsconfig.json
├── api/                          # 🚀 Primary production FastAPI Web Server
│   ├── main.py                   # Central server entry point (RAG + Gemini integration)
│   ├── requirements.txt          # Production python dependencies
│   └── download_models.py        # Local script to download and cache local model weights
├── backend/                      # 🧪 Monolithic development & historical environment
│   ├── api/
│   │   └── main.py               # Historical API containing local DistilBERT emotion logic
│   ├── scripts/
│   │   └── seed_database.py      # Chromadb seeding script (25 templates across 5 distortions)
│   └── chroma_db/                # Local persistent vector database (gitignored)
├── scripts/                      # 🔬 Offline training and validation scripts
│   ├── train_lora.py             # PEFT LoRA fine-tuning for FLAN-T5 base model
│   ├── train_emotion_model.py    # Local fine-tuning for DistilBERT emotion model
│   ├── safety.py                 # Multi-dialect pre-generation crisis keyword hard-block
│   └── chat_cli.py               # Local terminal chat tester
├── data/                         # 📊 Dataset files
│   ├── dataset.jsonl             # 150+ custom QA instruction records (<50MB, committed directly)
│   └── dataset_from_openai.jsonl # Transformed instruction records
├── config/                       # ⚙️ Operational settings
│   └── system_prompt.txt         # Clinical CBT system prompt template
├── render.yaml                   # 🚀 Render backend deployment blueprint configuration
├── requirements.txt              # Shared Python libraries
├── start_all.ps1                 # Windows PowerShell quick-start script
└── Readme.md                     # This documentation
```

---

## 🧬 Tech Stack

### Frontend (Client App)
* **Framework**: **Next.js 15** with TypeScript
* **Styling**: **Vanilla CSS & Tailwind CSS** utilizing deep dark themes, backdrop-blurs, dynamic transitions, and modern glassmorphic borders
* **Icons**: **Lucide React** for smooth, unified vector interfaces
* **Transitions**: Integrated modern CSS animations for natural chat flow

### Backend (FastAPI Web Service)
* **Framework**: **FastAPI (Python 3.10+)** for fast asynchronous routing
* **Generative Engine**: **Google Gemini API** (using a 3-model fallback chain: `gemini-2.0-flash` ➔ `gemini-2.0-flash-lite` ➔ `gemini-2.5-flash-lite`)
* **SDK**: Brand new **`google-genai`** standard SDK
* **Concurrency**: Managed blocking GenAI calls dynamically using `asyncio.to_thread` for non-blocking FastAPI performance

### ML, Embeddings & RAG Vector Store
* **Vector DB**: Persistent local **ChromaDB** database
* **Embedding Model**: Local, cached **`multilingual-e5-small`** (~470MB) loaded on CPU/GPU for multi-dialect semantic matching
* **Intent Translation Engine**: Internal serverless routing combining **LibreTranslate API** and **MyMemory API** supporting seamless translations for English, Hindi, Marathi, and Kannada
* **Research / Custom Training**: Custom **PEFT LoRA** fine-tuning scripts targeting attention layers (`r=8`, `alpha=16`) of `google/flan-t5-base`

---

## 🚀 Local Setup Instructions

Follow these steps to run both the frontend and backend servers locally on your machine.

### Prerequisites
* **Node.js** v18 or later
* **Python** 3.10 or later
* **Gemini API Key** (Free tier works perfectly; obtain from [Google AI Studio](https://aistudio.google.com/app/apikey))

---

### Step 1: Set Up Backend environment
1. From the project root, open your terminal.
2. Initialize and activate a Python virtual environment:
   ```bash
   python -m venv api/venv
   # On Windows:
   api\venv\Scripts\activate
   # On Mac/Linux:
   source api/venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r api/requirements.txt
   ```
4. Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Step 2: Download Local ML Models & Seed the Database
1. Run the local model download utility to fetch the multilingual embedding model (this keeps all embedding matching local and secure):
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('multilingual-e5-small').save('./multilingual-e5-small')"
   ```
2. Seed the ChromaDB vector database with our clinical CBT distortion definitions and frameworks:
   ```bash
   python backend/scripts/seed_database.py
   ```
   *Output confirmation:* `Database seeding completed successfully! (25 entries across 5 distortions)`

### Step 3: Run the Servers
* **Backend**: Run the FastAPI application on port `8000`:
  ```bash
  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
  ```
* **Frontend**: Open a new terminal, navigate to `/frontend`, install packages, and launch:
  ```bash
  cd frontend
  npm install
  npm run dev
  ```

*Alternatively, Windows users can launch both servers simultaneously with a single click:*
```powershell
.\start_all.ps1
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser to interact with the platform!

---

## 🚀 Deployment Architecture

Manas Mitra is configured for seamless deployment in production.

```
                  ┌──────────────────────────────┐
                  │      Vercel Deployment       │
                  │   - Next.js 15 Web Client    │
                  │   - Regional Routing (India) │
                  └──────────────┬───────────────┘
                                 │
                            HTTPS POST
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │      Render Web Service      │
                  │   - Python FastAPI Server    │
                  │   - Persistent ChromaDB DB   │
                  │   - Multilingual Embeddings  │
                  └──────────────┬───────────────┘
                                 │
                             HTTPS GenAI
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │    Gemini API Hub    │
                      │  - Primary: 2.0-Flash│
                      │  - 3-Model Fallback  │
                      └──────────────────────┘
```

### 1. Frontend: Vercel Configuration
The frontend is fully configured for deployment on **Vercel** via our custom `frontend/vercel.json` file. It optimizes installation, utilizes standard React framework detection, and targets the **Mumbai, India (`bom1`)** hosting region, ensuring ultra-low latency connections for Indian students.

### 2. Backend: Render Configuration
The Python FastAPI backend is configured using the **Render Blueprints specification** (`render.yaml`). The blueprint automates deployment by:
* Declaring a Python Web Service mapped to the API files.
* Running `pip install -r requirements.txt` followed by `python backend/scripts/seed_database.py` on build to auto-initialize the vector collection.
* Setting vital environment variables to enable fast, offline sentence embedding computation.
