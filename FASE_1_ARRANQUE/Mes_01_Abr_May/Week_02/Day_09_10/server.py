import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# 0. Load environment variables (this loads your GROQ_API_KEY from .env)
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(title="WhatsApp Webhook Server")

# 1. Initialize the LLM (Large Language Model)
# We do this OUTSIDE the webhook function so it only connects once when the server starts,
# instead of creating a new connection every single time a message arrives.
try:
    llm = ChatGroq(
        temperature=0.2,
        model="llama-3.1-8b-instant",
    )
    print("✅ Groq LLM initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Groq: {e}")
    llm = None

# 2. Create the memory "warehouse"
# This is a Python dictionary. Think of it like a filing cabinet:
# - Each "drawer" is labeled with a phone number (e.g. "whatsapp:+59169512710")
# - Inside each drawer is a LIST of messages (the full conversation history)
# When a new user writes, we create a new drawer for them automatically.
# When a known user writes again, we open their existing drawer and read their history.
conversation_memory: dict[str, list] = {}


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    # 1. Twilio sends data as form URL-encoded payload. We need to extract it.
    form_data = await request.form()
    
    # Extract the incoming message text and the sender's phone number
    body_field = form_data.get("Body", "")
    # FastAPI's form() can return strings or uploaded files. We ensure it's a string.
    incoming_msg = str(body_field).strip() if isinstance(body_field, str) else ""
    
    sender_number = form_data.get("From", "")
    
    print(f"📩 Received message from {sender_number}: '{incoming_msg}'")

    # 2. Prepare our response using Twilio's TwiML (XML format)
    response = MessagingResponse()
    
    # Basic logic: respond differently based on what they said
    if incoming_msg.lower() == "hello" or incoming_msg.lower() == "hola":
        reply_text = "Hi! I am your new local Python server. How can I help you?"
    else:
        reply_text = f"I received your message: '{incoming_msg}'. I'm not very smart yet, wait until Day 10!"

    # Add the text to the TwiML response
    response.message(reply_text)

    # 3. Return the response to Twilio as XML text
    return PlainTextResponse(str(response), media_type="application/xml")

# To run this server, we will use the command:
# uvicorn server:app --reload --port 8000
