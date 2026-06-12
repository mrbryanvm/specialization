"""
server.py — RAG + Lead Scoring + CRM + Outbound + Human Handoff
================================================================

Day 16: Evolution of Day 15's CRM bot.
NEW FEATURES:
  - Outbound Initiation: The bot STARTS conversations when a prospect fills a form.
    Make.com detects a new Google Form response → calls POST /api/outbound →
    we send a personalized WhatsApp greeting via Twilio REST API in <30 seconds.
  - Human Handoff: The bot detects when it can't help (keywords, repeated failures)
    and transfers to a human with the FULL conversation context. The owner is alerted
    via Make.com webhook. The bot stops responding until the owner resumes the chat.

ARCHITECTURE (Day 16):
    [Google Form] → Make.com → POST /api/outbound → Twilio → [User WhatsApp]
                                                                    ↓
    User replies → Twilio → POST /webhook → get_ai_response()
                                                ↓
                                   Handoff detection?
                                   YES → send_handoff_webhook() → Make.com → Owner alert
                                   NO  → RAG + Scoring + CRM (same as Day 15)

HOW TO RUN:
    1. Build the vector DB (once):  python chroma_helper.py
    2. Start the server:            uvicorn server:app --reload --port 8000
"""

import logging
import os
from datetime import datetime
from typing import Optional, cast

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pydantic import BaseModel, Field
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse

# ──────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MuebleriaBot")

app = FastAPI(
    title="Mueblería América — AI Sales Assistant v5",
    description=(
        "WhatsApp AI agent with RAG catalog search, lead scoring, "
        "CRM integration, OUTBOUND initiation from forms, and "
        "intelligent HUMAN HANDOFF with full context."
    ),
    version="5.0.0",
)

# ──────────────────────────────────────────────────────────────
# 1. LOAD THE VECTOR DATABASE (ChromaDB)
# ──────────────────────────────────────────────────────────────

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "catalogo_muebleria_america"

# HF_TOKEN is your HuggingFace API token (free).
# Instead of loading the 300MB model INTO this server's RAM,
# we send text to HuggingFace's servers and they return the vector.
# Same model, same quality, ~5MB RAM instead of ~300MB.
HF_TOKEN = os.getenv("HF_TOKEN", "")

try:
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model=EMBEDDING_MODEL,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    logger.info(f"✅ ChromaDB loaded from {CHROMA_DB_DIR}")
except Exception as e:
    logger.error(f"❌ Failed to load ChromaDB: {e}")
    retriever = None

# ──────────────────────────────────────────────────────────────
# 2. PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────


class LeadData(BaseModel):
    """Structured data extracted from a sales conversation (same as Day 14-15)."""

    nombre: Optional[str] = Field(
        default=None,
        description="Nombre del cliente si lo mencionó en la conversación.",
    )
    telefono: Optional[str] = Field(
        default=None,
        description="Número de teléfono si lo compartió.",
    )
    presupuesto: Optional[str] = Field(
        default=None,
        description="Presupuesto mencionado por el cliente (ej: '4000 Bs').",
    )
    producto_interes: Optional[str] = Field(
        default=None,
        description="Producto o categoría de interés (ej: 'sofá de cuero').",
    )
    timeline: Optional[str] = Field(
        default=None,
        description="Urgencia o plazo mencionado (ej: 'antes del sábado').",
    )
    score: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Puntaje de calificación del lead de 1 a 10.",
    )
    score_label: str = Field(
        default="COLD",
        description="Etiqueta: 'HOT' (8-10), 'WARM' (5-7), 'COLD' (1-4).",
    )
    resumen: str = Field(
        default="Conversación recién iniciada.",
        description="Resumen de 1 línea de la conversación hasta el momento.",
    )


class OutboundRequest(BaseModel):
    """
    ── NEW IN DAY 16 ──
    Data received from Make.com when a prospect fills the Google Form.
    Make.com calls POST /api/outbound with this JSON body.

    Fields:
        nombre:   Customer's name from the form
        telefono: Customer's phone number (just digits, e.g. "59169512710")
        interes:  What they're looking for (e.g. "muebles de sala")
    """

    nombre: str = Field(description="Nombre del prospecto del formulario.")
    telefono: str = Field(
        description=(
            "Teléfono del prospecto (solo dígitos, sin +, sin 'whatsapp:'). "
            "Ej: '59169512710' para Bolivia."
        )
    )
    interes: str = Field(
        default="muebles",
        description="Qué busca el prospecto (extraído del formulario).",
    )


# ──────────────────────────────────────────────────────────────
# 3. SYSTEM PROMPTS — same as Day 15 + handoff instruction
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres el asistente virtual de Mueblería América, una tienda de muebles
ubicada en Av. América #1245, Cochabamba, Bolivia. Tu nombre es "Ana" (Asistente de
Negocios Automatizado).

TU ROL:
- Eres una vendedora amable, profesional y directa.
- Ayudas a los clientes a encontrar el mueble perfecto según sus necesidades y presupuesto.
- Respondes SIEMPRE en español. Máximo 6 líneas por mensaje.
- Usas emojis con moderación (1-2 por mensaje, no más).

REGLAS CRÍTICAS DE RESPUESTA:
1. SOLO responde con información del CONTEXTO DEL CATÁLOGO proporcionado abajo.
2. Si la información NO está en el catálogo, di EXACTAMENTE:
   "Esa información no la tengo disponible en este momento. ¿Te gustaría que un asesor humano te ayude con eso? 😊"
3. NUNCA inventes precios, productos o características que no estén en el catálogo.
4. Cuando menciones un producto, SIEMPRE incluye: nombre, precio, y una característica clave.
5. Si el cliente pregunta algo general (saludo), responde brevemente y redirige a cómo puedes ayudarle.

TRANSFERENCIA A HUMANO:
- Si el cliente dice "quiero hablar con una persona", "agente", "humano", "queja" o
  expresa frustración → responde: "Entiendo, déjame conectarte con un especialista que
  tiene todo el contexto de nuestra conversación. Te contactará en menos de 5 minutos. 🙏"
- NO sigas respondiendo sobre productos después de este mensaje.

ESTRATEGIA DE CALIFICACIÓN (preguntas NATURALES, máximo 1 por mensaje):
- Si muestran interés pero no dijeron QUÉ buscan → pregunta qué tipo de mueble necesitan.
- Si dijeron QUÉ pero no CUÁNTO → pregunta el presupuesto naturalmente.
- Si dijeron QUÉ y CUÁNTO pero no CUÁNDO → pregunta el plazo.
- Si están decididos → ofrece visita al showroom o delivery.

CONTEXTO DEL CATÁLOGO (información real del negocio):
{catalog_context}
"""

SCORING_SYSTEM_PROMPT = """Eres un sistema de análisis de leads de ventas. Analiza la conversación
y extrae datos estructurados del cliente.

CRITERIOS DE SCORING:
- ¿Mencionó presupuesto específico? → +2 puntos
- ¿El presupuesto está dentro del rango (500-15000 Bs)? → +1 punto
- ¿Tiene timeline definido? → +2 puntos
- ¿Preguntó por productos específicos? → +2 puntos
- ¿Compartió su nombre? → +1 punto
- ¿Pidió visita, cotización o delivery? → +2 puntos

CLASIFICACIÓN: HOT (8-10) | WARM (5-7) | COLD (1-4)

REGLAS:
- Basa el score SOLO en lo que el cliente dijo explícitamente.
- Si acaba de saludar, score es 1 (COLD), campos son null.
- score_label DEBE ser exactamente "HOT", "WARM" o "COLD" (mayúsculas).
"""

# ──────────────────────────────────────────────────────────────
# 4. LLM INITIALIZATION — same as Day 15
# ──────────────────────────────────────────────────────────────

try:
    llm = ChatGroq(temperature=0.3, model="llama-3.1-8b-instant")
    scoring_llm = ChatGroq(temperature=0.1, model="llama-3.1-8b-instant")
    structured_scoring_llm = scoring_llm.with_structured_output(LeadData)
    logger.info("✅ Groq LLMs initialized (conversational + scoring).")
except Exception as e:
    logger.error(f"❌ Error initializing Groq LLMs: {e}")
    llm = None
    structured_scoring_llm = None

# ──────────────────────────────────────────────────────────────
# 5. TWILIO CLIENT — NEW IN DAY 16
#    Used for OUTBOUND messages (WE initiate the conversation).
#    Day 12-15 only used TwiML (replying to inbound messages).
#    Day 16 adds the Twilio REST Client to SEND messages proactively.
# ──────────────────────────────────────────────────────────────

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

try:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("✅ Twilio REST Client initialized for outbound messaging.")
except Exception as e:
    logger.error(f"❌ Error initializing Twilio client: {e}")
    twilio_client = None

# ──────────────────────────────────────────────────────────────
# 6. MAKE.COM WEBHOOK CONFIGURATION
#    MAKE_WEBHOOK_URL     → CRM (leads → Google Sheets + email)  [Day 15]
#    MAKE_HANDOFF_WEBHOOK_URL → Handoff alerts to owner          [Day 16 NEW]
# ──────────────────────────────────────────────────────────────

MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "")
MAKE_HANDOFF_WEBHOOK_URL = os.getenv("MAKE_HANDOFF_WEBHOOK_URL", "")

if MAKE_WEBHOOK_URL:
    logger.info(f"✅ CRM webhook configured: {MAKE_WEBHOOK_URL[:50]}...")
else:
    logger.warning("⚠️ MAKE_WEBHOOK_URL not set — CRM integration disabled.")

if MAKE_HANDOFF_WEBHOOK_URL:
    logger.info(f"✅ Handoff webhook configured: {MAKE_HANDOFF_WEBHOOK_URL[:50]}...")
else:
    logger.warning(
        "⚠️ MAKE_HANDOFF_WEBHOOK_URL not set — Handoff alerts will be logged only."
    )

# ──────────────────────────────────────────────────────────────
# 7. IN-MEMORY STATE STORAGE
#    All dicts are keyed by phone number (e.g. "whatsapp:+59169512710")
# ──────────────────────────────────────────────────────────────

# Chat history per phone number
conversation_memory: dict[str, list] = {}
conversation_timestamps: dict[str, str] = {}

# Lead qualification data per phone number
lead_tracker: dict[str, dict] = {}

# Which leads have been sent to the CRM webhook (deduplication) [Day 15]
webhook_sent: dict[str, str] = {}  # phone → last score_label sent

# ── NEW IN DAY 16 ──

# Tracks conversations the bot INITIATED (via /api/outbound)
outbound_log: dict[str, dict] = {}  # phone → {nombre, interes, sent_at}

# Tracks conversations that have been handed off to a human
# True  → bot is SILENT (human is handling this)
# False → bot responds normally
handoff_active: dict[str, bool] = {}

# Counts how many times in a row the bot said "no tengo esa información"
# When this reaches 3, we trigger automatic handoff
no_info_count: dict[str, int] = {}

# The phrase the bot uses when it doesn't have catalog info.
# MUST match what's in SYSTEM_PROMPT to detect it.
NO_INFO_PHRASE = "Esa información no la tengo disponible en este momento"

# Keywords that trigger human handoff when the CUSTOMER says them
HANDOFF_KEYWORDS = [
    "quiero hablar con una persona",
    "hablar con un humano",
    "hablar con alguien",
    "agente humano",
    "asesor humano",
    "quiero un asesor",
    "quiero hablar con alguien",
    "persona real",
    "necesito hablar con alguien",
    "queja",
    "reclamo",
    "estoy molesto",
    "estoy frustrado",
    "no me ayudas",
    "esto no sirve",
    "mal servicio",
]

# ──────────────────────────────────────────────────────────────
# 8. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────


def search_catalog(query: str) -> str:
    """Search ChromaDB for catalog chunks relevant to the query (RAG retrieval)."""
    if retriever is None:
        return "No hay información del catálogo disponible en este momento."

    try:
        results = retriever.invoke(query)
        if not results:
            return "No se encontró información relevante en el catálogo."

        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"--- Resultado {i} ---\n{doc.page_content}")

        context = "\n\n".join(context_parts)
        logger.info(f"🔍 Retrieved {len(results)} chunks for query: '{query[:50]}'")
        return context

    except Exception as e:
        logger.error(f"❌ ChromaDB search error: {e}")
        return "Error al buscar en el catálogo."


def evaluate_lead(sender_number: str, chat_history: list) -> LeadData | None:
    """
    Lead scoring engine (same as Day 14-15).
    Analyzes the full conversation and returns structured lead data.
    """
    if structured_scoring_llm is None or not chat_history:
        return None

    try:
        transcript_lines = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                transcript_lines.append(f"CLIENTE: {msg.content}")
            elif isinstance(msg, AIMessage):
                transcript_lines.append(f"ASISTENTE: {msg.content}")

        transcript = "\n".join(transcript_lines)

        scoring_messages = [
            SystemMessage(content=SCORING_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Analiza la siguiente conversación y extrae los datos del lead:\n\n"
                    f"{transcript}\n\n"
                    f"Extrae: nombre, teléfono, presupuesto, producto_interes, timeline, "
                    f"score (1-10), score_label (HOT/WARM/COLD), y resumen (1 línea)."
                )
            ),
        ]

        lead_data = cast(LeadData, structured_scoring_llm.invoke(scoring_messages))

        emoji = {"HOT": "🔥", "WARM": "🟡", "COLD": "🔵"}.get(
            lead_data.score_label, "❓"
        )
        logger.info(
            f"{emoji} Lead Score for {sender_number}: "
            f"{lead_data.score}/10 ({lead_data.score_label}) — {lead_data.resumen}"
        )

        return lead_data

    except Exception as e:
        logger.error(f"❌ Lead scoring error for {sender_number}: {e}")
        return None


def send_lead_to_webhook(sender_number: str, lead_data: dict) -> None:
    """
    Send WARM/HOT lead data to Make.com CRM webhook (same as Day 15).
    Runs as a background task to avoid latency for the WhatsApp user.
    """
    if not MAKE_WEBHOOK_URL:
        return

    score_label = lead_data.get("score_label", "COLD")

    if score_label == "COLD":
        logger.info(f"🔵 Lead {sender_number} is COLD — not sending to CRM")
        return

    last_sent_label = webhook_sent.get(sender_number)
    if last_sent_label == score_label:
        logger.info(
            f"⏭️ Lead {sender_number} already sent as {score_label} — skipping"
        )
        return

    payload = {
        "telefono": sender_number,
        "nombre": lead_data.get("nombre"),
        "presupuesto": lead_data.get("presupuesto"),
        "producto_interes": lead_data.get("producto_interes"),
        "timeline": lead_data.get("timeline"),
        "score": lead_data.get("score"),
        "score_label": score_label,
        "resumen": lead_data.get("resumen"),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(MAKE_WEBHOOK_URL, json=payload)

        if response.status_code == 200:
            emoji = "🔥" if score_label == "HOT" else "🟡"
            logger.info(
                f"{emoji} ✅ Lead sent to CRM! {sender_number} → "
                f"{score_label} ({lead_data.get('score')}/10)"
            )
            webhook_sent[sender_number] = score_label
        else:
            logger.error(
                f"❌ CRM Webhook error: HTTP {response.status_code} — "
                f"{response.text[:200]}"
            )

    except httpx.TimeoutException:
        logger.error(f"⏰ CRM Webhook timeout for {sender_number}")
    except Exception as e:
        logger.error(f"❌ CRM Webhook error for {sender_number}: {e}")


def send_handoff_webhook(sender_number: str, lead_data: dict, conversation_summary: str) -> None:
    """
    ── NEW IN DAY 16 ──

    Sends an ESCALATION alert to Make.com when a human handoff is triggered.

    This function is called when:
    1. The customer explicitly asks for a human
    2. The bot fails to answer 3 times in a row (no_info_count reaches 3)

    The webhook payload tells Make.com to:
    → Alert the owner via email/WhatsApp with the customer's context
    → Include the full conversation summary so the owner can pick up immediately

    Note: This uses MAKE_HANDOFF_WEBHOOK_URL (different from the CRM webhook).
    If not configured, we just log the escalation locally.
    """
    logger.warning(
        f"🤝 HANDOFF TRIGGERED for {sender_number} — "
        f"Lead: {lead_data.get('nombre', 'Unknown')} | "
        f"Score: {lead_data.get('score', 0)}/10"
    )

    payload = {
        "tipo": "handoff",
        "telefono": sender_number,
        "nombre": lead_data.get("nombre"),
        "producto_interes": lead_data.get("producto_interes"),
        "presupuesto": lead_data.get("presupuesto"),
        "score": lead_data.get("score"),
        "score_label": lead_data.get("score_label"),
        "resumen_lead": lead_data.get("resumen"),
        "resumen_conversacion": conversation_summary,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accion_requerida": "Contactar al cliente en los próximos 5 minutos",
    }

    if not MAKE_HANDOFF_WEBHOOK_URL:
        logger.warning(
            f"⚠️ MAKE_HANDOFF_WEBHOOK_URL not configured — "
            f"Handoff logged locally only. Payload:\n{payload}"
        )
        return

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(MAKE_HANDOFF_WEBHOOK_URL, json=payload)

        if response.status_code == 200:
            logger.info(
                f"🤝 ✅ Handoff alert sent to owner for {sender_number}!"
            )
        else:
            logger.error(
                f"❌ Handoff webhook error: HTTP {response.status_code}"
            )

    except httpx.TimeoutException:
        logger.error(f"⏰ Handoff webhook timeout for {sender_number}")
    except Exception as e:
        logger.error(f"❌ Handoff webhook error for {sender_number}: {e}")


def check_handoff_keywords(message: str) -> bool:
    """
    ── NEW IN DAY 16 ──

    Check if the customer's message contains any handoff trigger keywords.
    Returns True if handoff should be triggered, False otherwise.

    This is a simple keyword-based detection. For production, you'd use
    a small classifier model for better accuracy.
    """
    message_lower = message.lower()
    for keyword in HANDOFF_KEYWORDS:
        if keyword in message_lower:
            logger.info(f"🤝 Handoff keyword detected: '{keyword}'")
            return True
    return False


def build_conversation_summary(chat_history: list) -> str:
    """
    Build a human-readable conversation summary for the handoff webhook.
    Formats the chat history as a readable transcript (last 10 messages max).
    """
    if not chat_history:
        return "Conversación sin mensajes previos."

    recent = chat_history[-10:]  # Last 10 messages only
    lines = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            lines.append(f"👤 CLIENTE: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"🤖 ANA: {msg.content[:100]}...")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 9. MAIN AI RESPONSE FUNCTION — Updated for Day 16
# ──────────────────────────────────────────────────────────────


def get_ai_response(
    sender_number: str,
    user_input: str,
    background_tasks: BackgroundTasks | None = None,
) -> str | None:
    """
    The AI brain: RAG + Lead Scoring + CRM + Human Handoff detection.

    Day 16 adds:
    - Pre-check: if handoff is active, check 'volver al bot' FIRST, then go silent
    - Post-check: detect handoff trigger keywords in user message
    - Post-check: detect if bot said "no tengo esa información" 3 times in a row
    - If handoff triggered: send escalation webhook and activate handoff mode

    Returns:
        str  → normal AI response (send to customer)
        None → handoff is active, bot should be COMPLETELY SILENT (no message)

    Flow:
    1. Check 'volver al bot' FIRST (even during handoff) → resume if yes
    2. Check if handoff is active → return None (silence) if yes
    3. Check customer message for handoff keywords
    4. SEARCH the catalog for relevant info (RAG Retrieval)
    5. BUILD the prompt with catalog context (RAG Augmentation)
    6. GENERATE the AI response (LLM 1 — conversational)
    7. Check if the AI response contains the "no info" phrase (auto-handoff trigger)
    8. SCORE THE LEAD (LLM 2 — structured output)
    9. SEND TO CRM (background task — Make.com webhook)
    """
    if llm is None:
        return (
            "Lo siento, estoy experimentando dificultades técnicas. "
            "Por favor intenta de nuevo en unos minutos. 🙏"
        )

    # ── STEP 0: CHECK 'volver al bot' FIRST — even during handoff ──
    # This MUST come before the handoff block, otherwise the customer
    # can never escape handoff mode (the bug we found in testing).
    resume_phrases = ["volver al bot", "volver al asistente"]
    if any(p in user_input.lower() for p in resume_phrases):
        if handoff_active.get(sender_number, False):
            handoff_active[sender_number] = False
            no_info_count[sender_number] = 0
            logger.info(f"✅ {sender_number} resumed bot conversation via message.")
            return (
                "¡Claro! Vuelvo a estar contigo. 😊 ¿En qué te puedo ayudar "
                "con nuestro catálogo de muebles?"
            )

    # ── PRE-CHECK: Is this conversation already handed off? ──
    # If handoff is active, the bot is COMPLETELY SILENT.
    # Returning None tells the webhook endpoint to send an empty TwiML
    # response — the customer sees NO message from the bot at all.
    # The owner is handling this conversation manually from the WhatsApp app.
    if handoff_active.get(sender_number, False):
        logger.info(f"🤝 {sender_number} is in handoff mode — bot is COMPLETELY SILENT.")
        return None

    # Initialize memory for new users
    if sender_number not in conversation_memory:
        conversation_memory[sender_number] = []
        conversation_timestamps[sender_number] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        no_info_count[sender_number] = 0
        logger.info(f"🆕 New conversation started with {sender_number}")

    user_history = conversation_memory[sender_number]

    # ── HANDOFF CHECK: Did the customer ask for a human? ──
    if check_handoff_keywords(user_input):
        # Activate handoff BEFORE generating AI response
        handoff_active[sender_number] = True
        no_info_count[sender_number] = 0

        # Get current lead data for the handoff webhook
        current_lead = lead_tracker.get(sender_number, {}).get(
            "data",
            {"nombre": None, "score": 0, "score_label": "COLD", "resumen": "Sin datos previos"},
        )
        conversation_summary = build_conversation_summary(user_history)

        if background_tasks:
            background_tasks.add_task(
                send_handoff_webhook, sender_number, current_lead, conversation_summary
            )
        else:
            send_handoff_webhook(sender_number, current_lead, conversation_summary)

        handoff_msg = (
            "Entiendo, déjame conectarte con un especialista que tiene todo el "
            "contexto de nuestra conversación. Te contactará en menos de 5 minutos. 🙏"
        )

        # Save the exchange to memory even though it's a handoff
        user_history.append(HumanMessage(content=user_input))
        user_history.append(AIMessage(content=handoff_msg))

        logger.info(f"🤝 Handoff activated for {sender_number} via keyword detection")
        return handoff_msg

    # ── STEP 1: RETRIEVAL — Search the catalog ──
    catalog_context = search_catalog(user_input)

    # ── STEP 2: AUGMENTATION — Build prompt with retrieved context ──
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    try:
        # ── STEP 3: GENERATION — LLM responds using the catalog context ──
        response = chain.invoke(
            {
                "catalog_context": catalog_context,
                "history": user_history,
                "input": user_input,
            }
        )

        if isinstance(response.content, str):
            ai_text = response.content
        else:
            ai_text = str(response.content)

        # Save the exchange to memory
        user_history.append(HumanMessage(content=user_input))
        user_history.append(AIMessage(content=ai_text))

        # Cap history at 20 messages (10 pairs)
        if len(user_history) > 20:
            conversation_memory[sender_number] = user_history[-20:]

        logger.info(f"🤖 → {sender_number}: '{ai_text[:80]}...'")

        # ── STEP 4 (NEW): AUTO-HANDOFF — Did the bot say "no tengo info"? ──
        # We count how many consecutive times the bot hits the "no info" fallback.
        # If it happens 3 times in a row, we trigger automatic handoff.
        if NO_INFO_PHRASE in ai_text:
            no_info_count[sender_number] = no_info_count.get(sender_number, 0) + 1
            logger.info(
                f"⚠️ Bot hit 'no info' response for {sender_number} "
                f"({no_info_count[sender_number]}/3)"
            )

            if no_info_count[sender_number] >= 3:
                logger.warning(
                    f"🤝 Auto-handoff triggered for {sender_number} "
                    f"(bot couldn't answer 3 times)"
                )
                handoff_active[sender_number] = True
                no_info_count[sender_number] = 0

                current_lead = lead_tracker.get(sender_number, {}).get(
                    "data",
                    {
                        "nombre": None,
                        "score": 0,
                        "score_label": "COLD",
                        "resumen": "Sin datos previos",
                    },
                )
                conversation_summary = build_conversation_summary(user_history)

                if background_tasks:
                    background_tasks.add_task(
                        send_handoff_webhook,
                        sender_number,
                        current_lead,
                        conversation_summary,
                    )
                else:
                    send_handoff_webhook(
                        sender_number, current_lead, conversation_summary
                    )

                # Override the AI response with the handoff message
                ai_text = (
                    "Veo que tengo algunas limitaciones para responder tus preguntas. "
                    "Te conectaré con un especialista humano que podrá ayudarte mejor. "
                    "Te contactará en menos de 5 minutos. 🙏"
                )
        else:
            # Reset the counter if the bot gave a useful answer
            no_info_count[sender_number] = 0

        # ── STEP 5: LEAD SCORING — extract structured data ──
        lead_data = evaluate_lead(sender_number, user_history)
        if lead_data:
            lead_dict = lead_data.model_dump()
            lead_tracker[sender_number] = {
                "data": lead_dict,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message_count": len(user_history),
            }

            # ── STEP 6: CRM WEBHOOK — send to Make.com in the background ──
            if background_tasks:
                background_tasks.add_task(
                    send_lead_to_webhook, sender_number, lead_dict
                )
            else:
                send_lead_to_webhook(sender_number, lead_dict)

        return ai_text

    except Exception as e:
        logger.error(f"❌ Groq API error for {sender_number}: {e}")
        return (
            "Disculpa, estoy teniendo problemas técnicos en este momento. "
            "Por favor intenta de nuevo en unos minutos. 🙏"
        )


# ──────────────────────────────────────────────────────────────
# 10. TWILIO OUTBOUND SENDER — NEW IN DAY 16
# ──────────────────────────────────────────────────────────────


def send_whatsapp_outbound(
    to_number: str, nombre: str, interes: str
) -> dict:
    """
    ── NEW IN DAY 16 ──

    Sends the FIRST WhatsApp message to a prospect (outbound initiation).

    KEY DIFFERENCE from Day 12-15:
    - Days 12-15: We only used TwiML (XML response to Twilio's HTTP request).
      The user writes FIRST, Twilio calls our /webhook, we respond.
    - Day 16: We use the Twilio REST Client to INITIATE the conversation.
      We call Twilio's API directly to send a message to a phone number.

    IMPORTANT: The Twilio WhatsApp Sandbox only allows outbound messages to
    numbers that have already sent the "join" keyword to the sandbox.
    For production with real clients, you'd need an approved WhatsApp Business number.

    Args:
        to_number: Phone number with country code, e.g. "59169512710"
        nombre:    Prospect's name from the Google Form
        interes:   What they're looking for, e.g. "muebles de sala"

    Returns:
        dict with success/error status
    """
    if twilio_client is None:
        return {
            "success": False,
            "error": "Twilio client not initialized. Check TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN.",
        }

    # Format the phone number for WhatsApp
    # Twilio requires the "whatsapp:" prefix and "+" before the country code
    if not to_number.startswith("+"):
        to_number_formatted = f"whatsapp:+{to_number}"
    else:
        to_number_formatted = f"whatsapp:{to_number}"

    # Build the personalized greeting message
    greeting = (
        f"¡Hola {nombre}! 👋 Vi que te interesaron nuestros {interes}.\n\n"
        f"Soy Ana, el asistente virtual de Mueblería América. "
        f"Estoy aquí para ayudarte a encontrar el mueble perfecto para tu hogar.\n\n"
        f"¿Qué tipo de mueble estás buscando exactamente? 😊"
    )

    try:
        message = twilio_client.messages.create(
            body=greeting,
            from_=TWILIO_WHATSAPP_FROM,
            to=to_number_formatted,
        )

        # Initialize conversation memory so when they reply, context is ready
        conversation_memory[to_number_formatted] = []
        conversation_timestamps[to_number_formatted] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        no_info_count[to_number_formatted] = 0

        # Add the greeting to memory as an AI message so the bot remembers
        conversation_memory[to_number_formatted].append(
            AIMessage(content=greeting)
        )

        # Log the outbound contact
        outbound_log[to_number_formatted] = {
            "nombre": nombre,
            "interes": interes,
            "mensaje_sid": message.sid,
            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info(
            f"📤 Outbound WhatsApp sent to {to_number_formatted} "
            f"(SID: {message.sid}) — Prospect: {nombre}"
        )

        return {
            "success": True,
            "message_sid": message.sid,
            "to": to_number_formatted,
            "prospect": nombre,
        }

    except Exception as e:
        logger.error(
            f"❌ Failed to send outbound WhatsApp to {to_number_formatted}: {e}"
        )
        return {
            "success": False,
            "error": str(e),
            "to": to_number_formatted,
        }


# ──────────────────────────────────────────────────────────────
# 11. FASTAPI ENDPOINTS
# ──────────────────────────────────────────────────────────────


@app.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives incoming WhatsApp messages from Twilio.

    IMPORTANT (Day 16 fix):
    If get_ai_response() returns None, it means handoff is active.
    In that case, we return an EMPTY TwiML response — the bot sends
    NO message at all. This way the owner can chat with the customer
    from the WhatsApp app without the bot interrupting.
    """
    form_data = await request.form()

    body_field = form_data.get("Body", "")
    incoming_msg = str(body_field).strip() if isinstance(body_field, str) else ""

    sender_number = str(form_data.get("From", ""))

    logger.info(f"📩 ← {sender_number}: '{incoming_msg}'")

    if incoming_msg:
        reply_text = get_ai_response(sender_number, incoming_msg, background_tasks)
    else:
        reply_text = (
            "¡Recibí tu archivo! Por el momento solo puedo procesar texto. "
            "¿En qué puedo ayudarte? 😊"
        )

    twiml_response = MessagingResponse()

    # ── HANDOFF SILENCE: If reply_text is None, bot stays completely quiet ──
    # An empty <Response></Response> tells Twilio to NOT send any message.
    # The customer sees nothing from the bot. The human handles it.
    if reply_text is not None:
        twiml_response.message(reply_text)
    else:
        logger.info(f"🤫 Sending empty TwiML for {sender_number} (handoff silence)")

    return PlainTextResponse(str(twiml_response), media_type="application/xml")


@app.post("/api/outbound")
async def outbound_initiation(data: OutboundRequest):
    """
    ── NEW IN DAY 16 ──

    Called by Make.com when a prospect fills the Google Form.
    Sends a personalized WhatsApp greeting to the prospect.

    Expected JSON body (sent by Make.com):
    {
        "nombre": "Carlos Mamani",
        "telefono": "59169512710",
        "interes": "mesa de comedor de madera"
    }

    Make.com setup:
    1. Trigger: Google Forms → Watch Responses
    2. Action: HTTP → Make a Request → POST to this endpoint
    3. Body: map the form fields to nombre, telefono, interes
    """
    logger.info(
        f"📋 Outbound request received: {data.nombre} | "
        f"{data.telefono} | '{data.interes}'"
    )

    # Send the outbound WhatsApp message directly
    # (fast enough to not need background — Twilio responds in ~1s)
    result = send_whatsapp_outbound(data.telefono, data.nombre, data.interes)

    if result["success"]:
        return {
            "status": "✅ Outbound WhatsApp sent!",
            "prospect": data.nombre,
            "phone": data.telefono,
            "message_sid": result.get("message_sid"),
            "note": (
                "The conversation memory has been initialized. "
                "When the prospect replies, the bot will have context."
            ),
        }
    else:
        return {
            "status": "❌ Failed to send outbound WhatsApp",
            "error": result.get("error"),
            "prospect": data.nombre,
            "phone": data.telefono,
        }


@app.post("/api/resume/{phone_number}")
async def resume_conversation(phone_number: str):
    """
    ── NEW IN DAY 16 ──

    Allows the owner to DEACTIVATE the handoff and return the customer to the bot.

    Call this endpoint after the human agent has finished handling the conversation
    and the customer can continue with the bot.

    Usage:
        POST /api/resume/59169512710
        or
        POST /api/resume/whatsapp:+59169512710

    The phone number can be with or without the 'whatsapp:+' prefix.
    """
    # Normalize the phone number format
    if not phone_number.startswith("whatsapp:"):
        if not phone_number.startswith("+"):
            phone_number = f"whatsapp:+{phone_number}"
        else:
            phone_number = f"whatsapp:{phone_number}"

    if handoff_active.get(phone_number, False):
        handoff_active[phone_number] = False
        no_info_count[phone_number] = 0
        logger.info(f"✅ Handoff deactivated for {phone_number} by owner")
        return {
            "status": "✅ Handoff deactivated",
            "phone": phone_number,
            "message": (
                "The bot will now respond again to this customer. "
                "The next message from the customer will resume the bot conversation."
            ),
        }
    else:
        return {
            "status": "ℹ️ No active handoff found for this number",
            "phone": phone_number,
        }


@app.get("/")
async def health_check():
    """Root endpoint showing server status, including outbound and handoff stats."""
    active_handoffs = [
        phone for phone, active in handoff_active.items() if active
    ]

    return {
        "status": "🟢 Online",
        "bot_name": "Mueblería América — AI Sales Assistant",
        "version": "5.0.0 (RAG + Lead Scoring + CRM + Outbound + Handoff)",
        "model": "llama-3.1-8b-instant",
        "vector_db": "ChromaDB loaded" if retriever else "⚠️ ChromaDB NOT loaded",
        "scoring_llm": "Active" if structured_scoring_llm else "⚠️ NOT loaded",
        "crm_webhook": "✅ Connected" if MAKE_WEBHOOK_URL else "⚠️ NOT configured",
        "handoff_webhook": (
            "✅ Connected" if MAKE_HANDOFF_WEBHOOK_URL else "⚠️ NOT configured (alerts logged only)"
        ),
        "twilio_client": "✅ Ready" if twilio_client else "⚠️ NOT initialized",
        # Conversation stats
        "active_conversations": len(conversation_memory),
        "leads_tracked": len(lead_tracker),
        "leads_sent_to_crm": len(webhook_sent),
        # Day 16 stats
        "outbound_sent": len(outbound_log),
        "active_handoffs": len(active_handoffs),
        "handoff_numbers": active_handoffs,
        # Lead breakdown
        "leads_summary": {
            "hot": sum(
                1 for lt in lead_tracker.values() if lt["data"]["score_label"] == "HOT"
            ),
            "warm": sum(
                1 for lt in lead_tracker.values() if lt["data"]["score_label"] == "WARM"
            ),
            "cold": sum(
                1 for lt in lead_tracker.values() if lt["data"]["score_label"] == "COLD"
            ),
        },
        "conversations": {
            number: {
                "messages": len(history),
                "started_at": conversation_timestamps.get(number, "unknown"),
                "lead": lead_tracker.get(number, {}).get("data", "Not scored yet"),
                "sent_to_crm": webhook_sent.get(number, "Not sent"),
                "handoff_active": handoff_active.get(number, False),
                "is_outbound": number in outbound_log,
            }
            for number, history in conversation_memory.items()
        },
    }


@app.get("/leads")
async def get_leads(label: str | None = None):
    """View all scored leads, optionally filtered by label (HOT/WARM/COLD)."""
    leads = lead_tracker

    if label:
        label_upper = label.upper()
        leads = {
            number: data
            for number, data in leads.items()
            if data["data"]["score_label"] == label_upper
        }

    sorted_leads = dict(
        sorted(leads.items(), key=lambda x: x[1]["data"]["score"], reverse=True)
    )

    return {
        "total_leads": len(sorted_leads),
        "filter": label.upper() if label else "ALL",
        "leads": sorted_leads,
    }


@app.get("/handoffs")
async def get_handoffs():
    """
    ── NEW IN DAY 16 ──
    View all conversations currently in human handoff mode.
    Use POST /api/resume/{phone} to return a customer to the bot.
    """
    active = {
        phone: {
            "active": True,
            "lead": lead_tracker.get(phone, {}).get("data", "No lead data"),
            "messages": len(conversation_memory.get(phone, [])),
            "started_at": conversation_timestamps.get(phone, "unknown"),
        }
        for phone, is_active in handoff_active.items()
        if is_active
    }

    return {
        "total_active_handoffs": len(active),
        "handoffs": active,
        "tip": "Use POST /api/resume/{phone_number} to return a customer to the bot",
    }


@app.get("/outbound")
async def get_outbound_log():
    """
    ── NEW IN DAY 16 ──
    View all outbound conversations initiated from Google Forms.
    """
    return {
        "total_outbound": len(outbound_log),
        "outbound_contacts": outbound_log,
    }


@app.get("/test")
async def test_rag(q: str = "¿Qué productos tienen?"):
    """Test the full pipeline from browser. Example: /test?q=Hola+soy+Carlos"""
    test_number = "test_user_browser"
    response = get_ai_response(test_number, q)
    raw_context = search_catalog(q)
    lead_info = lead_tracker.get(test_number, {}).get("data", "Not scored yet")
    crm_status = webhook_sent.get(test_number, "Not sent to CRM yet")
    handoff_status = handoff_active.get(test_number, False)

    return {
        "question": q,
        "ai_response": response,
        "lead_data": lead_info,
        "crm_status": crm_status,
        "handoff_active": handoff_status,
        "retrieved_context": raw_context,
        "debug_info": {
            "chunks_retrieved": raw_context.count("--- Resultado"),
            "conversation_messages": len(conversation_memory.get(test_number, [])),
            "no_info_count": no_info_count.get(test_number, 0),
        },
    }


@app.delete("/reset/{phone_number}")
async def reset_conversation(phone_number: str):
    """Reset a conversation and all its data. Useful for testing."""
    cleared = []
    for store_name, store in [
        ("conversation_memory", conversation_memory),
        ("conversation_timestamps", conversation_timestamps),
        ("lead_tracker", lead_tracker),
        ("webhook_sent", webhook_sent),
        ("handoff_active", handoff_active),
        ("no_info_count", no_info_count),
        ("outbound_log", outbound_log),
    ]:
        if phone_number in store:
            del store[phone_number]
            cleared.append(store_name)

    if cleared:
        return {"status": f"Cleared {', '.join(cleared)} for {phone_number}"}
    else:
        return {"status": f"No data found for {phone_number}"}


@app.post("/test-webhook")
async def test_webhook_endpoint():
    """Send a fake HOT lead to Make.com CRM webhook to test the integration."""
    if not MAKE_WEBHOOK_URL:
        return {
            "status": "❌ MAKE_WEBHOOK_URL not configured in .env",
            "instructions": "Add MAKE_WEBHOOK_URL=https://hook.us2.make.com/your_url to your .env file",
        }

    fake_lead = {
        "telefono": "whatsapp:+59100000000",
        "nombre": "Test Lead (Prueba Day 16)",
        "presupuesto": "5000 Bs",
        "producto_interes": "Mesa de Comedor Milano",
        "timeline": "Esta semana",
        "score": 9,
        "score_label": "HOT",
        "resumen": "Lead de prueba — Day 16 Outbound + Handoff",
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(MAKE_WEBHOOK_URL, json=fake_lead)

        return {
            "status": "✅ Webhook test sent!",
            "http_status": response.status_code,
            "response": response.text[:200] if response.text else "No response body",
            "payload_sent": fake_lead,
        }
    except Exception as e:
        return {
            "status": f"❌ Webhook test failed: {e}",
            "payload_attempted": fake_lead,
        }


@app.post("/test-outbound")
async def test_outbound_endpoint():
    """
    ── NEW IN DAY 16 ──
    Send a test outbound WhatsApp to YOUR phone number to verify the Twilio integration.
    Uses USER_NUMBER from .env as the destination.

    Usage: curl -X POST http://localhost:8000/test-outbound
    """
    user_number = os.getenv("USER_NUMBER", "")
    if not user_number:
        return {
            "status": "❌ USER_NUMBER not configured in .env",
            "instructions": "Add USER_NUMBER=whatsapp:+59169512710 to your .env file",
        }

    # Strip the 'whatsapp:+' prefix if present to pass just the digits
    digits = user_number.replace("whatsapp:+", "").replace("whatsapp:", "").replace("+", "")

    result = send_whatsapp_outbound(
        to_number=digits,
        nombre="Bryan (Test)",
        interes="muebles de sala y comedor",
    )

    return {
        "status": "✅ Test outbound sent!" if result["success"] else "❌ Test failed",
        "result": result,
        "note": (
            "Check your WhatsApp. You should receive a personalized greeting message. "
            "When you reply, the bot will respond with full context."
        ),
    }
