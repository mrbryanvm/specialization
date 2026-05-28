import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from twilio.twiml.messaging_response import MessagingResponse

# 0. Load environment variables from .env (GROQ_API_KEY, Twilio credentials, etc.)
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(title="WhatsApp AI Webhook Server")

# 1. Initialize the LLM once when the server starts.
# Doing this outside the webhook means we don't reconnect to Groq on every message.
# If the API key is wrong or missing, llm is set to None and we handle it gracefully.
try:
    llm = ChatGroq(
        temperature=0.2,
        model="llama-3.1-8b-instant",
    )
    print("✅ Groq LLM initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Groq LLM: {e}")
    llm = None

# 2. The memory "warehouse".
# A plain Python dictionary: keys are phone numbers, values are lists of messages.
# Example after two users write:
# {
#   "whatsapp:+59169512710": [HumanMessage("Hi"), AIMessage("Hello!")],
#   "whatsapp:+1234567890":  [HumanMessage("Hola"), AIMessage("¡Hola!")],
# }
conversation_memory: dict[str, list] = {}


def get_ai_response(sender_number: str, user_input: str) -> str:
    """
    3. The AI brain function.
    Takes who sent the message and what they said.
    Returns the AI's reply as a plain string.
    """
    # If the LLM never loaded, return an error message instead of crashing.
    if llm is None:
        return (
            "System Error: AI model is not initialized. Please check the server logs."
        )

    # If this phone number is new, create an empty history list for them.
    if sender_number not in conversation_memory:
        conversation_memory[sender_number] = []

    # Grab this specific user's chat history.
    user_history = conversation_memory[sender_number]

    # 3a. Define the prompt structure.
    # "system" sets the persona and rules for the AI.
    # MessagesPlaceholder injects the full chat history into the prompt automatically.
    # "human" is the new message the user just sent.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are 'BoImport Expert', a specialized assistant for Bolivian
entrepreneurs importing tech hardware from China.
Give practical, direct advice about logistics, customs, and suppliers.
Keep answers under 6 lines.
ONLY answer questions about international trade, logistics, or hardware.
If asked about anything else, politely decline and redirect to your specialty.""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # 3b. Build the LCEL chain: prompt → llm.
    # The | (pipe) operator connects them. The output of the prompt feeds into the llm.
    chain = prompt | llm

    try:
        # 3c. Invoke the chain. Pass the user's history and their new message.
        response = chain.invoke(
            {
                "history": user_history,
                "input": user_input,
            }
        )
        # Assert/ensure that ai_text is a string to satisfy the type checker (Pyright)
        if isinstance(response.content, str):
            ai_text = response.content
        else:
            ai_text = str(response.content)

        # 3d. Save both messages to memory AFTER a successful response.
        # We always save in pairs: what the human said → what the AI replied.
        user_history.append(HumanMessage(content=user_input))
        user_history.append(AIMessage(content=ai_text))

        # 3e. Cap the history at 20 messages (10 pairs) to avoid hitting token limits.
        # Slicing [-20:] keeps only the 20 most recent messages.
        if len(user_history) > 20:
            conversation_memory[sender_number] = user_history[-20:]

        print(f"🤖 AI replied to {sender_number}: '{ai_text[:60]}...'")
        return ai_text

    except Exception as e:
        # If Groq is down or the API key fails mid-conversation, reply politely.
        print(f"❌ Error calling Groq API: {e}")
        return "I'm sorry, I'm having technical difficulties right now. Please try again in a moment."


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    # 4. Extract the incoming data from Twilio's POST request.
    form_data = await request.form()

    body_field = form_data.get("Body", "")
    incoming_msg = str(body_field).strip() if isinstance(body_field, str) else ""

    sender_number = str(form_data.get("From", ""))

    print(f"📩 Received message from {sender_number}: '{incoming_msg}'")

    # 5. Decide what to reply.
    if incoming_msg:
        # There is text → send it to the AI brain.
        reply_text = get_ai_response(sender_number, incoming_msg)
    else:
        # No text → user sent a sticker, image, or audio.
        reply_text = "I received your media! I can only process text for now. Please type your question."

    # 6. Wrap the reply in Twilio's TwiML format and send it back.
    response = MessagingResponse()
    response.message(reply_text)

    return PlainTextResponse(str(response), media_type="application/xml")


# To run this server:
# uvicorn server:app --reload --port 8000
