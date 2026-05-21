# import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# Load environment variables from .env file (only works locally)
load_dotenv()

# --- PAGE SETUP ---
st.set_page_config(page_title="BoImport Expert", page_icon="📦")
st.title("📦 BoImport Logistics Assistant")
st.write("Professional interface for tech hardware import logistics.")


def initialize_session():
    """
    Initializes session state using modern LCEL architecture.
    - 'messages': stores the full chat history for UI rendering AND as memory for the LLM.
    - 'llm': the language model, stored once to avoid re-instantiation on every rerun.
    """
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(
            temperature=0.2,
            model="llama-3.1-8b-instant",
            # api_key=os.getenv("GROQ_API_KEY"),
        )

    if "messages" not in st.session_state:
        # This list acts as BOTH the visual chat history AND the LLM's memory.
        # Each item is a LangChain message object (HumanMessage or AIMessage).
        st.session_state.messages = []


def get_ai_response(user_input: str) -> str:
    """
    Builds the prompt chain and gets a response using the modern LCEL approach.
    No ConversationChain, no deprecated classes.
    """
    # 1. Define the prompt structure with a system persona and chat history placeholder
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are 'BoImport Expert', a specialized consultant for Bolivian
        entrepreneurs importing tech hardware from China.
        Provide technical, practical, and direct advice about logistics and customs.
        Keep answers under 6 lines. Use Markdown for clarity.
        ONLY answer questions related to international trade, logistics, and hardware.
        If asked about unrelated topics, politely decline and redirect to logistics.""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # 2. Build the chain: prompt | llm (LCEL pipe syntax)
    chain = prompt | st.session_state.llm

    # 3. Invoke the chain, passing the full message history as memory
    response = chain.invoke({"history": st.session_state.messages, "input": user_input})

    return response.content


# --- APP MAIN FLOW ---
initialize_session()

# Render existing chat history as bubbles
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle new user input
if user_prompt := st.chat_input("Ask about shipping, taxes, or suppliers..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get AI response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Processing logistics query..."):
            try:
                ai_text = get_ai_response(user_prompt)
                st.markdown(ai_text)
            except Exception as e:
                ai_text = f"System Error: {str(e)}"
                st.error(ai_text)

    # Save both messages to history AFTER displaying them
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    st.session_state.messages.append(AIMessage(content=ai_text))
