import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available Embedding Models for your API Key:")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(f" - {m.name}")

print("Available Chat Models for your API Key:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f" - {m.name}")