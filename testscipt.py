import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load the key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use EXACT name from your list
MODEL = "models/gemini-2.5-flash"
model = genai.GenerativeModel(MODEL)

resp = model.generate_content("Hello Gemini! Please confirm you are working.")
print(resp.text)
