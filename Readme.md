# Manas Mitra - Mental Health Chatbot

Manas Mitra is an AI-powered mental health assistant designed to provide empathetic and supportive conversations. It utilizes advanced Natural Language Processing (NLP) to detect user emotions and generates tailored responses to help users feel heard and understood.

## 🌟 Overview

The core mission of Manas Mitra is to offer a safe, judgment-free space for users to express their feelings. By combining local deep learning models for emotion recognition with the generative capabilities of Google's Gemini AI, the chatbot bridges the gap between technical accuracy and human-like empathy.

## ✨ Features

- **Real-time Emotion Detection**: Analyzes user input to identify emotions such as *Sadness, Joy, Love, Anger, Fear,* and *Surprise* using the `distilbert-base-uncased-emotion` model.
- **Empathetic AI Responses**: Generates context-aware, supportive, and concise responses using Google's **Gemini Flash** model.
- **FastAPI Backend**: robust and high-performance API handling requests asynchronously.
- **Privacy-Focused**: Runs emotion detection locally (or on the server) before querying the LLM, ensuring a dedicated processing pipeline.

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **ML/AI**:
  - `torch` & `transformers` (Hugging Face) for Emotion Detection
  - `google-generativeai` for Response Generation (Gemini API)
- **Server**: Uvicorn

### Frontend
- **Framework**: React / Next.js (Assumed based on structure)
- **Styling**: Modern CSS / Tailwind (Review project for specifics)

## 🚀 How It Works

1. **Input**: The user types a message in the chat interface.
2. **Analysis**: The backend receives the message and passes it through a **BERT-based Emotion Classifier** to determine the user's emotional state.
3. **Context Construction**: A prompt is constructed combining the *User's Message* and the *Detected Emotion* (e.g., "The user is feeling sad").
4. **Generation**: This prompt is sent to the **Gemini API**, which acts as a compassionate mental health assistant to generate a reply.
5. **Response**: The empathetic reply is sent back to the frontend and displayed to the user.

## 📦 Setup & Installation

### Prerequisites
- Python 3.9+
- Node.js & npm
- A Google Cloud API Key for Gemini

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Backend Setup
Navigate to the backend directory:
```bash
cd backend
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install dependencies:
```bash
pip install -r api/requirements.txt
```
*Note: If you need to run the heavy ML dependencies, check `requirements.txt` in the root of backend as well.*

**Environment Variables**:
Create a `.env` file in `backend/api/` or `backend/` and add your Gemini API key:
```ini
GEMINI_API_KEY=your_actual_api_key_here
```

Run the Server:
```bash
python api/main.py
# OR
uvicorn api.main:app --reload
```
The backend will start at `http://localhost:8000`.

### 3. Frontend Setup
Navigate to the frontend directory:
```bash
cd ../frontend
```

Install dependencies:
```bash
npm install
```

Run the application:
```bash
npm start
# OR
npm run dev
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
[Include License Here, e.g., MIT]
