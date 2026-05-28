import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from twilio.twiml.messaging_response import MessagingResponse

# ──────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────

# Load environment variables from .env
load_dotenv()

# Set up professional logging instead of just print()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("InmoBot")

# Initialize the FastAPI application
app = FastAPI(
    title="InmoBot Cochabamba — AI Real Estate Assistant",
    description="WhatsApp AI agent that qualifies real estate leads instantly.",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────
# 1. THE SYSTEM PROMPT — This is the "brain" of your product.
#    Change this single block to sell the SAME code to a
#    completely different industry.
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres 'InmoBot', el asistente virtual de una inmobiliaria en Cochabamba, 
Bolivia. Eres experto en el mercado inmobiliario cochabambino y tu trabajo es ser 
GENUINAMENTE ÚTIL para las personas que te escriben.

TU PERSONALIDAD:
- Eres como un amigo que sabe mucho de bienes raíces en Cochabamba.
- Hablas de forma natural, cálida y directa. Nada de frases corporativas vacías.
- Si alguien te insulta o se frustra, responde con humor y paciencia, no te ofendas.
- Responde SIEMPRE en español. Máximo 6 líneas por mensaje.

LO QUE SÍ PUEDES Y DEBES HACER:
1. Dar información REAL sobre rangos de precios por zona (usa los datos de abajo).
2. Explicar los trámites para comprar, alquilar o vender en Bolivia.
3. Recomendar zonas según el presupuesto y necesidades del cliente.
4. Comparar zonas (ventajas y desventajas de cada una).
5. Explicar qué documentos se necesitan (CI, comprobante de ingresos, etc.).
6. Dar consejos prácticos (qué revisar antes de alquilar, banderas rojas, etc.).
7. Responder preguntas generales relacionadas con vivienda y trámites.

CONOCIMIENTO DE PRECIOS (rangos aproximados 2024-2026):

ALQUILER mensual:
- Zona Norte (Sarco, Cala Cala, Queru Queru): Deptos $350-800/mes, Casas $600-1500/mes
- Tiquipaya: Deptos $200-450/mes, Casas $350-800/mes (condominios nuevos)
- Sacaba: Deptos $150-350/mes, Casas $250-600/mes (más accesible, familias jóvenes)
- Centro: Deptos $200-500/mes, Locales comerciales $300-1200/mes
- Quillacollo: Deptos $120-300/mes, Casas $200-500/mes (económico, en expansión)
- Zona Sur (Mesadilla, Valle Hermoso): Deptos $180-400/mes, Casas $300-700/mes

COMPRA:
- Zona Norte: Deptos $60,000-180,000, Casas $120,000-350,000+
- Tiquipaya: Deptos $40,000-90,000, Casas $70,000-200,000
- Sacaba: Deptos $30,000-70,000, Casas $50,000-150,000
- Centro: Deptos $35,000-100,000, Locales $50,000-250,000
- Quillacollo: Deptos $25,000-60,000, Casas $40,000-120,000
- Terrenos: Desde $15/m² (zonas alejadas) hasta $300/m² (Zona Norte)

TRÁMITES COMUNES:
- Para ALQUILAR: CI vigente, comprobante de ingresos, depósito (generalmente 2 meses), 
  contrato de alquiler (idealmente notariado). Algunos piden garante.
- Para COMPRAR: CI, NIT, minuta de compraventa, pago de impuestos de transferencia (3%), 
  registro en Derechos Reales, certificado catastral. Trámite total: 30-60 días.
- Para VENDER/OFERTAR: Título de propiedad, pago de impuestos al día, certificado 
  catastral, planos aprobados, CI del propietario.

CÓMO ACTUAR:
- Si preguntan precios → DA los rangos de precios de arriba según la zona.
- Si preguntan por trámites → EXPLICA los pasos claramente.
- Si preguntan por zonas → RECOMIENDA según su presupuesto y necesidades.
- Si ya tienes suficiente info sobre lo que buscan → ofrece agendar una visita 
  o conectarlos con un asesor para ver opciones específicas.
- Si preguntan algo NO relacionado con vivienda/inmuebles → intenta ayudar brevemente 
  si puedes, y luego redirige amablemente.
- NUNCA digas "no puedo ayudarte con eso". Siempre intenta dar algo útil.

EJEMPLO:
Cliente: "Cuánto cuesta alquilar en zona norte?"
Tú: "En Zona Norte los deptos van desde $350 hasta $800/mes dependiendo del tamaño. 
Un 2 dormitorios bien ubicado en Cala Cala ronda los $450-550. Las casas arrancan 
en $600. ¿Tienes un presupuesto en mente? Así te ubico mejor 😊"
"""

# ──────────────────────────────────────────────────────────────
# 2. LLM INITIALIZATION
# ──────────────────────────────────────────────────────────────

try:
    llm = ChatGroq(
        temperature=0.3,  # Slightly more creative for natural Spanish conversation
        model="llama-3.1-8b-instant",
    )
    logger.info("✅ Groq LLM (Llama 3.1 8B) initialized successfully.")
except Exception as e:
    logger.error(f"❌ Error initializing Groq LLM: {e}")
    llm = None

# ──────────────────────────────────────────────────────────────
# 3. CONVERSATIONAL MEMORY
#    Each phone number gets its own isolated chat history.
# ──────────────────────────────────────────────────────────────

conversation_memory: dict[str, list] = {}

# Track when each conversation started (for logging/analytics)
conversation_timestamps: dict[str, str] = {}


def get_ai_response(sender_number: str, user_input: str) -> str:
    """
    The AI brain. Takes a phone number + message, returns the AI's reply.
    Manages per-user memory automatically.
    """
    if llm is None:
        return (
            "Lo siento, estoy experimentando dificultades técnicas. "
            "Por favor intenta de nuevo en unos minutos. 🙏"
        )

    # Initialize memory for new users
    if sender_number not in conversation_memory:
        conversation_memory[sender_number] = []
        conversation_timestamps[sender_number] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        logger.info(f"🆕 New conversation started with {sender_number}")

    user_history = conversation_memory[sender_number]

    # Build the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # LCEL chain: prompt → llm
    chain = prompt | llm

    try:
        response = chain.invoke(
            {
                "history": user_history,
                "input": user_input,
            }
        )

        # Type-safe extraction of the AI's text
        if isinstance(response.content, str):
            ai_text = response.content
        else:
            ai_text = str(response.content)

        # Save the exchange to memory
        user_history.append(HumanMessage(content=user_input))
        user_history.append(AIMessage(content=ai_text))

        # Cap history at 20 messages (10 pairs) to avoid token limits
        if len(user_history) > 20:
            conversation_memory[sender_number] = user_history[-20:]

        logger.info(f"🤖 → {sender_number}: '{ai_text[:80]}...'")
        return ai_text

    except Exception as e:
        logger.error(f"❌ Groq API error for {sender_number}: {e}")
        return (
            "Disculpa, estoy teniendo problemas técnicos en este momento. "
            "Por favor intenta de nuevo en unos minutos. 🙏"
        )


# ──────────────────────────────────────────────────────────────
# 4. WEBHOOK ENDPOINT
#    Twilio POSTs here every time someone sends a WhatsApp message.
# ──────────────────────────────────────────────────────────────


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()

    # Extract the message text safely
    body_field = form_data.get("Body", "")
    incoming_msg = str(body_field).strip() if isinstance(body_field, str) else ""

    # Extract the sender's phone number
    sender_number = str(form_data.get("From", ""))

    logger.info(f"📩 ← {sender_number}: '{incoming_msg}'")

    # Route to AI or handle media
    if incoming_msg:
        reply_text = get_ai_response(sender_number, incoming_msg)
    else:
        reply_text = (
            "¡Recibí tu archivo! Por el momento solo puedo procesar texto. "
            "¿En qué puedo ayudarte? 😊"
        )

    # Build and return the TwiML response
    twiml_response = MessagingResponse()
    twiml_response.message(reply_text)

    return PlainTextResponse(str(twiml_response), media_type="application/xml")


# ──────────────────────────────────────────────────────────────
# 5. HEALTH CHECK & STATS ENDPOINT
#    Useful for monitoring and demo purposes.
# ──────────────────────────────────────────────────────────────


@app.get("/")
async def health_check():
    """Root endpoint showing server status and basic stats."""
    return {
        "status": "🟢 Online",
        "bot_name": "InmoBot Cochabamba",
        "model": "llama-3.1-8b-instant",
        "active_conversations": len(conversation_memory),
        "conversations": {
            number: {
                "messages": len(history),
                "started_at": conversation_timestamps.get(number, "unknown"),
            }
            for number, history in conversation_memory.items()
        },
    }


# ──────────────────────────────────────────────────────────────
# HOW TO RUN:
#   cd into the Day_12 folder, then:
#   uvicorn server:app --reload --port 8000
# ──────────────────────────────────────────────────────────────
