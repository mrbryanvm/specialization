"""
server.py — RAG-Powered WhatsApp AI Sales Assistant with Lead Scoring + CRM
=============================================================================

Day 15: Evolution of Day 14's Lead Scoring bot.
NEW FEATURES:
  - CRM Integration: Lead data is sent to Make.com via webhook
  - Google Sheets: Make.com automatically logs leads in a Google Sheet
  - Owner Alerts: HOT leads (score >= 8) trigger an email alert to the owner
  - Background Tasks: Webhook calls run in the background (no latency for user)

ARCHITECTURE:
    User sends WhatsApp → Twilio → FastAPI /webhook
        ↓
    get_ai_response(sender, message)
        ↓
    Step 1: Search ChromaDB for top 4 relevant chunks (RAG)
    Step 2: Build prompt with retrieved context + conversation history
    Step 3: LLM generates conversational response
    Step 4: LEAD SCORING — a second LLM call extracts structured data
    Step 5: CRM WEBHOOK — send lead data to Make.com (background task)
        ↓
    Response sent back via Twilio → WhatsApp
    Lead data updated in memory + sent to Google Sheets

HOW TO RUN:
    1. First, build the vector DB:  python chroma_helper.py
    2. Then start the server:       uvicorn server:app --reload --port 8000
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
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
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
    title="Mueblería América — AI Sales Assistant with RAG + Lead Scoring + CRM",
    description=(
        "WhatsApp AI agent that searches the real business catalog, "
        "qualifies leads with intelligent scoring (1-10), and automatically "
        "logs them in Google Sheets with owner alerts for HOT leads."
    ),
    version="4.0.0",
)

# ──────────────────────────────────────────────────────────────
# 1. LOAD THE VECTOR DATABASE (ChromaDB)
# ──────────────────────────────────────────────────────────────

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "catalogo_muebleria_america"

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
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
# 2. PYDANTIC MODEL FOR LEAD DATA (Structured Output)
#    The LLM will output a JSON object matching this schema.
# ──────────────────────────────────────────────────────────────


class LeadData(BaseModel):
    """Structured data extracted from a sales conversation."""

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
        description=(
            "Presupuesto mencionado por el cliente "
            "(ej: '4000 Bs', 'entre 2000 y 3000 Bs')."
        ),
    )
    producto_interes: Optional[str] = Field(
        default=None,
        description=(
            "Producto o categoría de interés "
            "(ej: 'mesa de comedor de madera', 'sofá de cuero')."
        ),
    )
    timeline: Optional[str] = Field(
        default=None,
        description=(
            "Urgencia o plazo mencionado "
            "(ej: 'antes del sábado', 'esta semana', 'no tengo prisa')."
        ),
    )
    score: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Puntaje de calificación del lead de 1 a 10. "
            "Criterios: presupuesto específico (+2), presupuesto dentro del rango "
            "del negocio (+1), timeline definido (+2), producto específico "
            "(+2), nombre compartido (+1), pidió visita/cotización (+2)."
        ),
    )
    score_label: str = Field(
        default="COLD",
        description="Etiqueta: 'HOT' (8-10), 'WARM' (5-7), 'COLD' (1-4).",
    )
    resumen: str = Field(
        default="Conversación recién iniciada.",
        description="Resumen de 1 línea de la conversación hasta el momento.",
    )


# ──────────────────────────────────────────────────────────────
# 3. SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────

# 3a. CONVERSATIONAL prompt (for the customer-facing response)
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
2. Si la información NO está en el catálogo, di: "Esa información no la tengo disponible
   en este momento. ¿Te gustaría que un asesor humano te ayude con eso? 😊"
3. NUNCA inventes precios, productos o características que no estén en el catálogo.
4. Cuando menciones un producto, SIEMPRE incluye: nombre, precio, y una característica clave.
5. Si el cliente pregunta algo general (saludo, cómo estás), responde brevemente y
   redirige a cómo puedes ayudarle con muebles.

ESTRATEGIA DE CALIFICACIÓN (haz estas preguntas de forma NATURAL, no robótica):
- Si el cliente muestra interés pero NO ha dicho qué busca → pregunta: "¿Qué tipo de
  mueble estás buscando?" o "¿Es para algún espacio en particular?"
- Si ya dijo qué busca pero NO mencionó presupuesto → pregunta naturalmente:
  "¿Tienes un presupuesto en mente? Así te muestro las mejores opciones 😊"
- Si ya dijo producto y presupuesto pero NO timeline → pregunta:
  "¿Para cuándo lo necesitarías? Pregunto por el tema de disponibilidad y delivery."
- Si el cliente está decidido → ofrece coordinar una visita al showroom o el delivery.
- Si el cliente duda → menciona las promociones vigentes o el financiamiento.

IMPORTANTE: No hagas TODAS las preguntas de golpe. Máximo UNA pregunta de calificación
por mensaje, mezclada con la respuesta sobre productos. Debe sentirse como una conversación
natural, NO como un formulario.

CONTEXTO DEL CATÁLOGO (información real del negocio):
{catalog_context}
"""

# 3b. SCORING prompt (for the lead extraction — internal, not shown to user)
SCORING_SYSTEM_PROMPT = """Eres un sistema de análisis de leads de ventas. Tu trabajo es
analizar una conversación entre un cliente y un asistente de ventas de una mueblería,
y extraer datos estructurados del cliente.

CRITERIOS DE SCORING (suma los puntos que apliquen):
- ¿El cliente mencionó un presupuesto específico? → +2 puntos
- ¿El presupuesto está dentro del rango del negocio (500-15000 Bs)? → +1 punto
- ¿Tiene una necesidad con timeline definido? (ej: "lo necesito esta semana",
  "antes del sábado") → +2 puntos
- ¿Preguntó por productos específicos, no solo "qué tienen"? → +2 puntos
- ¿Compartió su nombre voluntariamente? → +1 punto
- ¿Pidió agendar una visita, cotización formal o delivery? → +2 puntos

CLASIFICACIÓN:
- HOT: 8-10 puntos (listo para comprar, contactar al dueño inmediatamente)
- WARM: 5-7 puntos (interesado pero necesita más información o tiempo)
- COLD: 1-4 puntos (solo curioseando o conversación muy temprana)

REGLAS:
- Basa el score SOLO en lo que el cliente dijo explícitamente en la conversación.
- NO inventes información que el cliente no mencionó.
- Si el cliente acaba de saludar, el score es 1 (COLD) y los campos son null.
- El resumen debe ser de UNA sola línea, conciso y útil para un vendedor humano.
- score_label DEBE ser exactamente "HOT", "WARM" o "COLD" (en mayúsculas).
"""

# ──────────────────────────────────────────────────────────────
# 4. LLM INITIALIZATION
# ──────────────────────────────────────────────────────────────

try:
    # Main conversational LLM
    llm = ChatGroq(
        temperature=0.3,
        model="llama-3.1-8b-instant",
    )

    # Scoring LLM — uses structured output to return LeadData JSON
    # Lower temperature for more consistent, predictable scoring
    scoring_llm = ChatGroq(
        temperature=0.1,
        model="llama-3.1-8b-instant",
    )
    structured_scoring_llm = scoring_llm.with_structured_output(LeadData)

    logger.info("✅ Groq LLMs initialized (conversational + scoring).")
except Exception as e:
    logger.error(f"❌ Error initializing Groq LLMs: {e}")
    llm = None
    structured_scoring_llm = None

# ──────────────────────────────────────────────────────────────
# 5. MAKE.COM WEBHOOK CONFIGURATION (Day 15 — CRM Integration)
# ──────────────────────────────────────────────────────────────

MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "")

if MAKE_WEBHOOK_URL:
    logger.info(f"✅ Make.com webhook configured: {MAKE_WEBHOOK_URL[:50]}...")
else:
    logger.warning(
        "⚠️ MAKE_WEBHOOK_URL not set in .env — CRM integration disabled. "
        "Leads will only be stored in memory."
    )

# ──────────────────────────────────────────────────────────────
# 6. CONVERSATIONAL MEMORY + LEAD TRACKER
# ──────────────────────────────────────────────────────────────

# Chat history per phone number
conversation_memory: dict[str, list] = {}
conversation_timestamps: dict[str, str] = {}

# Lead qualification data per phone number
lead_tracker: dict[str, dict] = {}

# Track which leads have already been sent to the webhook
# to avoid sending duplicate entries for the same phone number
webhook_sent: dict[str, str] = {}  # phone_number → last score_label sent


def search_catalog(query: str) -> str:
    """
    Search the ChromaDB vector database for chunks relevant to the query.
    This is the RETRIEVAL step of RAG.
    """
    if retriever is None:
        logger.warning("⚠️ ChromaDB not available, returning empty context")
        return "No hay información del catálogo disponible en este momento."

    try:
        results = retriever.invoke(query)

        if not results:
            return "No se encontró información relevante en el catálogo."

        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"--- Resultado {i} ---\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        logger.info(f"🔍 Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return context

    except Exception as e:
        logger.error(f"❌ ChromaDB search error: {e}")
        return "Error al buscar en el catálogo."


def evaluate_lead(sender_number: str, chat_history: list) -> LeadData | None:
    """
    LEAD SCORING ENGINE — Day 14's core feature.

    Takes the full conversation history and uses a second LLM call
    with structured output (Pydantic) to extract:
    - Client name, budget, product interest, timeline
    - A score from 1-10
    - A label: HOT, WARM, or COLD

    This runs AFTER the conversational response is sent, so it
    doesn't add latency to the user experience.
    """
    if structured_scoring_llm is None:
        logger.warning("⚠️ Scoring LLM not available")
        return None

    if not chat_history:
        return None

    try:
        # Build the scoring prompt with the full conversation
        # We format the chat history as a readable transcript
        transcript_lines = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                transcript_lines.append(f"CLIENTE: {msg.content}")
            elif isinstance(msg, AIMessage):
                transcript_lines.append(f"ASISTENTE: {msg.content}")

        transcript = "\n".join(transcript_lines)

        # Build messages for the scoring LLM
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

        # Call the structured output LLM — returns a LeadData instance directly
        lead_data = cast(LeadData, structured_scoring_llm.invoke(scoring_messages))

        # Log the result
        emoji = {"HOT": "🔥", "WARM": "🟡", "COLD": "🔵"}.get(
            lead_data.score_label, "❓"
        )
        logger.info(
            f"{emoji} Lead Score for {sender_number}: "
            f"{lead_data.score}/10 ({lead_data.score_label}) — "
            f"{lead_data.resumen}"
        )

        return lead_data

    except Exception as e:
        logger.error(f"❌ Lead scoring error for {sender_number}: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 7. WEBHOOK SENDER — Day 15's core feature
#    Sends lead data to Make.com in the background.
# ──────────────────────────────────────────────────────────────


def send_lead_to_webhook(sender_number: str, lead_data: dict) -> None:
    """
    Send lead data to the Make.com webhook for CRM integration.

    This function runs as a FastAPI BackgroundTask, meaning it executes
    AFTER the HTTP response has been sent to the client. This ensures
    zero added latency for the WhatsApp user.

    The webhook payload includes all lead fields plus the phone number
    and a timestamp. Make.com will:
    1. Log the lead in Google Sheets (all WARM + HOT leads)
    2. Send an email alert to the owner (only HOT leads, score >= 8)

    DEDUPLICATION LOGIC:
    - We track what we've already sent in `webhook_sent`
    - A lead is re-sent to the webhook when:
      a) It's the first time we're sending this phone number, OR
      b) The lead's score_label has CHANGED (e.g., COLD → WARM, WARM → HOT)
    - This prevents spamming Make.com with the same data on every message
    """
    if not MAKE_WEBHOOK_URL:
        logger.warning("⚠️ Webhook URL not configured — skipping CRM push")
        return

    score_label = lead_data.get("score_label", "COLD")

    # Only send WARM (5-7) and HOT (8-10) leads to the CRM
    if score_label == "COLD":
        logger.info(
            f"🔵 Lead {sender_number} is COLD — not sending to CRM"
        )
        return

    # Deduplication: check if we already sent this lead with the same label
    last_sent_label = webhook_sent.get(sender_number)
    if last_sent_label == score_label:
        logger.info(
            f"⏭️ Lead {sender_number} already sent as {score_label} — skipping"
        )
        return

    # Build the payload for Make.com
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
        # Send the HTTP POST request to Make.com
        with httpx.Client(timeout=10.0) as client:
            response = client.post(MAKE_WEBHOOK_URL, json=payload)

        if response.status_code == 200:
            emoji = "🔥" if score_label == "HOT" else "🟡"
            logger.info(
                f"{emoji} ✅ Lead sent to CRM! {sender_number} → "
                f"{score_label} ({lead_data.get('score')}/10)"
            )
            # Mark as sent so we don't duplicate
            webhook_sent[sender_number] = score_label
        else:
            logger.error(
                f"❌ Webhook error: HTTP {response.status_code} — "
                f"{response.text[:200]}"
            )

    except httpx.TimeoutException:
        logger.error(
            f"⏰ Webhook timeout for {sender_number} — Make.com did not respond "
            f"in 10 seconds. Lead is still saved in memory."
        )
    except Exception as e:
        logger.error(f"❌ Webhook error for {sender_number}: {e}")


def get_ai_response(
    sender_number: str,
    user_input: str,
    background_tasks: BackgroundTasks | None = None,
) -> str:
    """
    The AI brain with RAG + Lead Scoring + CRM. For each message:
    1. SEARCH the catalog for relevant info (Retrieval)
    2. BUILD the prompt with that info (Augmentation)
    3. GENERATE the response (Generation)
    4. SCORE THE LEAD — extract structured data from the conversation
    5. SEND TO CRM — push lead data to Make.com webhook (background)
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

        # ── STEP 4: LEAD SCORING — extract structured data ──
        lead_data = evaluate_lead(sender_number, user_history)
        if lead_data:
            lead_dict = lead_data.model_dump()
            lead_tracker[sender_number] = {
                "data": lead_dict,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message_count": len(user_history),
            }

            # ── STEP 5: CRM WEBHOOK — send to Make.com in the background ──
            if background_tasks:
                background_tasks.add_task(
                    send_lead_to_webhook, sender_number, lead_dict
                )
            else:
                # If no background_tasks available (e.g. /test endpoint),
                # send synchronously
                send_lead_to_webhook(sender_number, lead_dict)

        return ai_text

    except Exception as e:
        logger.error(f"❌ Groq API error for {sender_number}: {e}")
        return (
            "Disculpa, estoy teniendo problemas técnicos en este momento. "
            "Por favor intenta de nuevo en unos minutos. 🙏"
        )


# ──────────────────────────────────────────────────────────────
# 8. WHATSAPP WEBHOOK ENDPOINT
# ──────────────────────────────────────────────────────────────


@app.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
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
    twiml_response.message(reply_text)

    return PlainTextResponse(str(twiml_response), media_type="application/xml")


# ──────────────────────────────────────────────────────────────
# 9. HEALTH CHECK & LEAD DASHBOARD ENDPOINT
# ──────────────────────────────────────────────────────────────


@app.get("/")
async def health_check():
    """Root endpoint showing server status, conversations, and lead scores."""
    return {
        "status": "🟢 Online",
        "bot_name": "Mueblería América — AI Sales Assistant",
        "version": "4.0.0 (RAG + Lead Scoring + CRM)",
        "model": "llama-3.1-8b-instant",
        "vector_db": "ChromaDB loaded" if retriever else "⚠️ ChromaDB NOT loaded",
        "scoring_llm": "Active" if structured_scoring_llm else "⚠️ NOT loaded",
        "crm_webhook": "✅ Connected" if MAKE_WEBHOOK_URL else "⚠️ NOT configured",
        "active_conversations": len(conversation_memory),
        "leads_tracked": len(lead_tracker),
        "leads_sent_to_crm": len(webhook_sent),
        "leads_summary": {
            "hot": sum(
                1 for lt in lead_tracker.values() if lt["data"]["score_label"] == "HOT"
            ),
            "warm": sum(
                1
                for lt in lead_tracker.values()
                if lt["data"]["score_label"] == "WARM"
            ),
            "cold": sum(
                1
                for lt in lead_tracker.values()
                if lt["data"]["score_label"] == "COLD"
            ),
        },
        "conversations": {
            number: {
                "messages": len(history),
                "started_at": conversation_timestamps.get(number, "unknown"),
                "lead": lead_tracker.get(number, {}).get("data", "Not scored yet"),
                "sent_to_crm": webhook_sent.get(number, "Not sent"),
            }
            for number, history in conversation_memory.items()
        },
    }


# ──────────────────────────────────────────────────────────────
# 10. LEAD DASHBOARD — View all scored leads
# ──────────────────────────────────────────────────────────────


@app.get("/leads")
async def get_leads(label: str | None = None):
    """
    View all scored leads, optionally filtered by label.
    Examples:
        GET /leads           → all leads
        GET /leads?label=HOT → only HOT leads
    """
    leads = lead_tracker

    if label:
        label_upper = label.upper()
        leads = {
            number: data
            for number, data in leads.items()
            if data["data"]["score_label"] == label_upper
        }

    # Sort by score descending (hottest first)
    sorted_leads = dict(
        sorted(
            leads.items(),
            key=lambda x: x[1]["data"]["score"],
            reverse=True,
        )
    )

    return {
        "total_leads": len(sorted_leads),
        "filter": label.upper() if label else "ALL",
        "leads": sorted_leads,
    }


# ──────────────────────────────────────────────────────────────
# 11. TEST ENDPOINT — Quick way to test RAG + Scoring + CRM
# ──────────────────────────────────────────────────────────────


@app.get("/test")
async def test_rag(q: str = "¿Qué productos tienen?"):
    """
    Test the RAG + Lead Scoring + CRM pipeline from the browser.
    Example: http://localhost:8000/test?q=Hola,+soy+Carlos.+Quiero+una+mesa+de+4000+Bs
    """
    test_number = "test_user_browser"
    response = get_ai_response(test_number, q)

    raw_context = search_catalog(q)

    lead_info = lead_tracker.get(test_number, {}).get("data", "Not scored yet")
    crm_status = webhook_sent.get(test_number, "Not sent to CRM yet")

    return {
        "question": q,
        "ai_response": response,
        "lead_data": lead_info,
        "crm_status": crm_status,
        "retrieved_context": raw_context,
        "debug_info": {
            "chunks_retrieved": raw_context.count("--- Resultado"),
            "conversation_messages": len(conversation_memory.get(test_number, [])),
            "webhook_configured": bool(MAKE_WEBHOOK_URL),
        },
    }


# ──────────────────────────────────────────────────────────────
# 12. RESET ENDPOINT — Clear a conversation for testing
# ──────────────────────────────────────────────────────────────


@app.delete("/reset/{phone_number}")
async def reset_conversation(phone_number: str):
    """
    Reset a conversation and its lead data.
    Useful for testing: DELETE /reset/test_user_browser
    """
    cleared = []
    if phone_number in conversation_memory:
        del conversation_memory[phone_number]
        cleared.append("conversation_memory")
    if phone_number in conversation_timestamps:
        del conversation_timestamps[phone_number]
        cleared.append("conversation_timestamps")
    if phone_number in lead_tracker:
        del lead_tracker[phone_number]
        cleared.append("lead_tracker")
    if phone_number in webhook_sent:
        del webhook_sent[phone_number]
        cleared.append("webhook_sent")

    if cleared:
        return {"status": f"Cleared {', '.join(cleared)} for {phone_number}"}
    else:
        return {"status": f"No data found for {phone_number}"}


# ──────────────────────────────────────────────────────────────
# 13. MANUAL WEBHOOK TEST — Send a fake lead to test Make.com
# ──────────────────────────────────────────────────────────────


@app.post("/test-webhook")
async def test_webhook_endpoint():
    """
    Send a fake HOT lead to the Make.com webhook to test the integration.
    Use this to verify that:
    1. Make.com receives the data
    2. Google Sheets gets a new row
    3. The owner email alert fires

    Usage: curl -X POST http://localhost:8000/test-webhook
    """
    if not MAKE_WEBHOOK_URL:
        return {
            "status": "❌ MAKE_WEBHOOK_URL not configured in .env",
            "instructions": (
                "Add MAKE_WEBHOOK_URL=https://hook.us1.make.com/your_url "
                "to your .env file"
            ),
        }

    fake_lead = {
        "telefono": "whatsapp:+59100000000",
        "nombre": "Test Lead (Prueba)",
        "presupuesto": "5000 Bs",
        "producto_interes": "Mesa de Comedor Milano",
        "timeline": "Esta semana",
        "score": 9,
        "score_label": "HOT",
        "resumen": "Lead de prueba para verificar la integración con Make.com",
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
            "next_steps": [
                "1. Check your Make.com scenario execution history",
                "2. Verify Google Sheets has a new row",
                "3. Check your email for the HOT lead alert",
            ],
        }
    except Exception as e:
        return {
            "status": f"❌ Webhook test failed: {e}",
            "payload_attempted": fake_lead,
        }


# ──────────────────────────────────────────────────────────────
# HOW TO RUN:
#   1. Build the DB:       python chroma_helper.py
#   2. Start server:       uvicorn server:app --reload --port 8000
#   3. Test in browser:    http://localhost:8000/test?q=Hola+soy+Carlos
#   4. View leads:         http://localhost:8000/leads
#   5. Test webhook:       curl -X POST http://localhost:8000/test-webhook
#   6. Reset test:         curl -X DELETE http://localhost:8000/reset/test_user_browser
# ──────────────────────────────────────────────────────────────
