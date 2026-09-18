import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chatbot_config import CHATBOT_NAME, SYSTEM_PROMPT

load_dotenv()
app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY","")) if os.getenv("GEMINI_API_KEY") else None
MODEL = "gemini-3.1-flash-lite"

@app.get("/")
def home():
    return render_template("index.html", chatbot_name=CHATBOT_NAME)

@app.post("/api/chat")
def chat():
    if not client:
        return jsonify({"error":"Gemini API key is not configured."}),500
    data=request.get_json(silent=True) or {}
    message=(data.get("message") or "").strip()
    history=data.get("history") or []
    if not message:
        return jsonify({"error":"Message is required."}),400
    contents=[]
    for item in history[-12:]:
        if item.get("role") in ("user","model") and item.get("text"):
            contents.append(types.Content(role=item["role"],parts=[types.Part.from_text(text=item["text"])]))
    contents.append(types.Content(role="user",parts=[types.Part.from_text(text=message)]))
    try:
        r=client.models.generate_content(
            model=MODEL,contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,temperature=0.3))
        return jsonify({"reply":r.text or "I could not generate a response."})
    except Exception:
        return jsonify({"error":"Unable to process the request right now."}),500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT","5000")))
