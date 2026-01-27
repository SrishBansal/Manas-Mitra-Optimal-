import os
import google.generativeai as genai
from dotenv import load_dotenv

# Force load .env from current directory
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key present: {bool(api_key)}")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

try:
    genai.configure(api_key=api_key)
    print("Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error occurred: {e}")
