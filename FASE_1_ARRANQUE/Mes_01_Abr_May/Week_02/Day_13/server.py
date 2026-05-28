"""
server.py — RAG-Powered WhatsApp AI Sales Assistant
====================================================

Day 13: This server is the evolution of Day 12's InmoBot.
The CRITICAL difference: instead of hardcoding business knowledge in the prompt,
the bot SEARCHES a vector database (ChromaDB) for relevant information and
responds with REAL catalog data.

ARCHITECTURE:
    User sends WhatsApp → Twilio → FastAPI /webhook
        ↓
    get_ai_response(sender, message)
        ↓
    Step 1: Search ChromaDB for top 3 relevant chunks
    Step 2: Build prompt with retrieved context + conversation history
    Step 3: LLM generates response using ONLY the catalog data
        ↓
    Response sent back via Twilio → WhatsApp

HOW TO RUN:
    1. First, build the vector DB:  python chroma_helper.py
    2. Then start the server:       uvicorn server:app --reload --port 8000
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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
    title="Mueblería América — AI Sales Assistant with RAG",
    description="WhatsApp AI agent that searches the real business catalog to answer customer questions.",
    version="2.0.0",
)

# ──────────────────────────────────────────────────────────────
# 1. LOAD THE VECTOR DATABASE (ChromaDB)
#    This is the catalog search engine. It was built by chroma_helper.py.
#    We load it ONCE when the server starts, then search it for every message.
# ──────────────────────────────────────────────────────────────

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "catalogo_muebleria_america"

try:
    # Load the same embedding model used to build the database
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    # Connect to the existing ChromaDB (read-only, no rebuild)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )

    # Create a retriever that fetches the top 4 most relevant chunks
    # k=4 means: for each question, find the 4 best matching pieces of catalog
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    logger.info(f"✅ ChromaDB loaded from {CHROMA_DB_DIR}")
    logger.info(f"   📊 Collection: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to load ChromaDB: {e}")
    logger.error("   Did you run 'python chroma_helper.py' first?")
    retriever = None

# ──────────────────────────────────────────────────────────────
# 2. THE SYSTEM PROMPT
#    Notice: NO product data here! The products come from ChromaDB.
#    This prompt only defines the bot's PERSONALITY and BEHAVIOR.
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
2. Si la información NO está en el catálogo, di: "Esa información no la tengo disponible 
   en este momento. ¿Te gustaría que un asesor humano te ayude con eso? 😊"
3. NUNCA inventes precios, productos o características que no estén en el catálogo.
4. Cuando menciones un producto, SIEMPRE incluye: nombre, precio, y una característica clave.
5. Si el cliente pregunta algo general (saludo, cómo estás), responde brevemente y 
   redirige a cómo puedes ayudarle con muebles.

ESTRATEGIA DE VENTA (actúa natural, no robótica):
- Si el cliente muestra interés en un producto → pregunta qué espacio tiene disponible
- Si el cliente pregunta por precios → menciona el producto y ofrece mostrar opciones similares
- Si el cliente está decidido → ofrece coordinar una visita al showroom o el delivery
- Si el cliente duda → menciona las promociones vigentes o el financiamiento

CONTEXTO DEL CATÁLOGO (información real del negocio):
{catalog_context}
"""

# ──────────────────────────────────────────────────────────────
# 3. LLM INITIALIZATION
# ──────────────────────────────────────────────────────────────

try:
    llm = ChatGroq(
        temperature=0.3,
        model="llama-3.1-8b-instant",
    )
    logger.info("✅ Groq LLM (Llama 3.1 8B) initialized successfully.")
except Exception as e:
    logger.error(f"❌ Error initializing Groq LLM: {e}")
    llm = None

# ──────────────────────────────────────────────────────────────
# 4. CONVERSATIONAL MEMORY
#    Each phone number gets its own isolated chat history.
# ──────────────────────────────────────────────────────────────

conversation_memory: dict[str, list] = {}
conversation_timestamps: dict[str, str] = {}


def search_catalog(query: str) -> str:
    """
    Search the ChromaDB vector database for chunks relevant to the query.

    This is the RETRIEVAL step of RAG:
    - Takes the customer's question
    - Finds the 4 most semantically similar chunks from the catalog
    - Returns them as a single formatted string

    Example:
        query = "¿Tienen sofás de cuero?"
        → Returns the Torino and Viena product entries from the catalog
    """
    if retriever is None:
        logger.warning("⚠️ ChromaDB not available, returning empty context")
        return "No hay información del catálogo disponible en este momento."

    try:
        # .invoke() performs the semantic search
        results = retriever.invoke(query)

        if not results:
            return "No se encontró información relevante en el catálogo."

        # Combine all retrieved chunks into one context string
        # Each chunk is separated by a divider for clarity
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"--- Resultado {i} ---\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        logger.info(f"🔍 Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return context

    except Exception as e:
        logger.error(f"❌ ChromaDB search error: {e}")
        return "Error al buscar en el catálogo."


def get_ai_response(sender_number: str, user_input: str) -> str:
    """
    The AI brain with RAG. For each message:
    1. SEARCH the catalog for relevant info (Retrieval)
    2. BUILD the prompt with that info (Augmentation)
    3. GENERATE the response (Generation)
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

    # LCEL chain: prompt → llm
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

        # Extract the AI's text
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
        return ai_text

    except Exception as e:
        logger.error(f"❌ Groq API error for {sender_number}: {e}")
        return (
            "Disculpa, estoy teniendo problemas técnicos en este momento. "
            "Por favor intenta de nuevo en unos minutos. 🙏"
        )


# ──────────────────────────────────────────────────────────────
# 5. WEBHOOK ENDPOINT
#    Twilio POSTs here every time someone sends a WhatsApp message.
# ──────────────────────────────────────────────────────────────


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()

    body_field = form_data.get("Body", "")
    incoming_msg = str(body_field).strip() if isinstance(body_field, str) else ""

    sender_number = str(form_data.get("From", ""))

    logger.info(f"📩 ← {sender_number}: '{incoming_msg}'")

    if incoming_msg:
        reply_text = get_ai_response(sender_number, incoming_msg)
    else:
        reply_text = (
            "¡Recibí tu archivo! Por el momento solo puedo procesar texto. "
            "¿En qué puedo ayudarte? 😊"
        )

    twiml_response = MessagingResponse()
    twiml_response.message(reply_text)

    return PlainTextResponse(str(twiml_response), media_type="application/xml")


# ──────────────────────────────────────────────────────────────
# 6. HEALTH CHECK & STATS ENDPOINT
# ──────────────────────────────────────────────────────────────


@app.get("/")
async def health_check():
    """Root endpoint showing server status and basic stats."""
    return {
        "status": "🟢 Online",
        "bot_name": "Mueblería América — AI Sales Assistant",
        "version": "2.0.0 (RAG)",
        "model": "llama-3.1-8b-instant",
        "vector_db": "ChromaDB loaded" if retriever else "⚠️ ChromaDB NOT loaded",
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
# 7. TEST ENDPOINT — Quick way to test RAG without WhatsApp
#    Send a GET request to /test?q=your+question
# ──────────────────────────────────────────────────────────────


@app.get("/test")
async def test_rag(q: str = "¿Qué productos tienen?"):
    """
    Test the RAG pipeline directly from the browser.
    Example: http://localhost:8000/test?q=¿Tienen+sofás+de+cuero?
    """
    # Use a test phone number for the conversation
    test_number = "test_user_browser"
    response = get_ai_response(test_number, q)

    # Also show the raw retrieved context for debugging
    raw_context = search_catalog(q)

    return {
        "question": q,
        "ai_response": response,
        "retrieved_context": raw_context,
        "debug_info": {
            "chunks_retrieved": raw_context.count("--- Resultado"),
            "conversation_messages": len(conversation_memory.get(test_number, [])),
        },
    }


# ──────────────────────────────────────────────────────────────
# HOW TO RUN:
#   1. Build the DB:     python chroma_helper.py
#   2. Start server:     uvicorn server:app --reload --port 8000
#   3. Test in browser:  http://localhost:8000/test?q=¿Tienen+mesas?
# ──────────────────────────────────────────────────────────────
