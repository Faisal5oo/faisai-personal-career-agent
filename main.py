import os
import json
import requests
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- INITIALIZATION ---
load_dotenv(override=True)

app = FastAPI(title="Faisal Haroon Personal Agent API")

# Enable CORS so your website frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Gemini client (for evaluation, currently commented out)
# gemini = OpenAI(
#     api_key=os.getenv("GOOGLE_API_KEY"), 
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

webhook_url = os.getenv("WEBHOOK_URL")

# --- DATA MODELS ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

# --- TOOL UTILITIES ---
def send_discord_message(message):
    if not webhook_url:
        print("Discord Webhook URL not set.")
        return
    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("Alert sent to Discord!")
    else:
        print(f"Failed Discord Alert: {response.status_code}")

def record_user_details(email, name="Name not provided", notes="Notes not provided"):
    send_discord_message(f"📩 **New Lead:**\nName: {name}\nEmail: {email}\nNotes: {notes}")
    return f"Details for {name} recorded successfully!"

def record_unknown_question(question):
    send_discord_message(f"❓ **Unknown Question:** {question}")
    return f"Question recorded for Faisal to review."

# Tool Definitions for LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Record interest from a user and their contact email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record a question that is outside your knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            }
        }
    }
]

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Executing tool: {tool_name}")
        
        if tool_name == "record_user_details":
            result = record_user_details(**arguments)
        elif tool_name == "record_unknown_question":
            result = record_unknown_question(**arguments)
        else:
            result = "Tool not found"
            
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results

# --- CONTEXT LOADING ---
# Ensuring files are loaded once on startup
linkedin = ""
try:
    reader = PdfReader("about-me/linkedin.pdf")
    for page in reader.pages:
        linkedin += page.extract_text() or ""
except Exception as e:
    print(f"Error loading PDF: {e}")

summary = ""
try:
    with open("about-me/my-profile-summary.txt", "r", encoding="UTF-8") as f:
        summary = f.read()
except Exception as e:
    print(f"Error loading summary: {e}")

name = "Faisal Haroon"
system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

tool_protocols = """
# TOOL CALLING RULES
1. **NEVER** talk and call a tool in the same turn. If calling a tool, the response must be ONLY the tool call.
2. **NO HALLUCINATIONS:** If an answer isn't in the provided ## Summary or ## LinkedIn, you MUST use `record_unknown_question`.
3. **LEAD CAPTURE (record_user_details):**
   - TRIGGER: User shows project interest or prepares to leave (bye/thanks).
   - STEP 1: Ask for Name and Email in the same response and record them.
   - STEP 2: If Name and Email are provided, call the tool to record details otherwise do not call the record_user_details tool.
   - STEP 3: Call tool ONLY when both are provided.
   - STEP 4: This is the main rule only ask for thr user details when he is interested for a project or any question that is unknown and not in the provided context or either the user is leaving by saying the leaving words (for example Thanks , Thank you, Bye, nice talking to you).
   - FORBIDDEN: Do not use placeholders like "null" or "user@example.com please make sure to call the tool only when the user provides both. ".
"""

system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."
system_prompt += tool_protocols

# --- EVALUATION CODE (COMMENTED OUT) ---
"""
class Evaluation(BaseModel):
    is_acceptable : bool
    feedback : str

def evaluate(reply, message, history) -> Evaluation:
    # Logic for Gemini evaluation
    pass
"""

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "running", "agent": name}

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        # Construct message history for the LLM
        # Converting incoming history to the format LLM expects
        formatted_history = [{"role": m.role, "content": m.content} for m in payload.history]
        
        messages = [
            {"role": "system", "content": system_prompt},
            *formatted_history,
            {"role": "user", "content": payload.message}
        ]

        done = False
        final_reply = ""

        while not done:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", # or llama-3.1-8b-instant
                messages=messages,
                tools=tools
            )
            
            resp_message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls":
                messages.append(resp_message)
                tool_results = handle_tool_calls(resp_message.tool_calls)
                messages.extend(tool_results)
                # Continue loop to let model respond to tool result
            else:
                final_reply = resp_message.content
                done = True

        return {"response": final_reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)