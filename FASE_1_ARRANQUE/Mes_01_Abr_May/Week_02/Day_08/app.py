import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables for API keys
load_dotenv()

def initialize_session():
    """
    Initializes the session state to prevent 'memory loss' during Streamlit reruns.
    This pattern ensures the LLM and History are only instantiated once.
    """
    if "bot" not in st.session_state:
        # Optimization: Define the model once and store it in session state
        llm = ChatGroq(
            temperature=0.2, 
            model_name="llama-3.1-8b-instant", 
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # System Prompt: Strict persona for BoImport Expert
        template = """
        You are 'BoImport Expert', a specialized consultant for hardware importers.
        Provide technical, practical, and direct advice regarding logistics and customs.
        Keep answers under 6 lines. Use Markdown for clarity.
        
        Current conversation:
        {history}
        User: {input}
        BoImport Expert:
        """
        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        
        # Storing the Chain in session_state preserves the ConversationBufferMemory
        st.session_state.bot = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=ConversationBufferMemory()
        )
        
        # History for UI rendering
        st.session_state.messages = []

def main():
    st.set_page_config(page_title="BoImport Expert", page_icon="📦")
    st.title("📦 BoImport Logistics Assistant")
    st.write("Professional interface for tech hardware import logistics.")

    initialize_session()

    # --- RENDER CHAT HISTORY ---
    # We iterate through the session list to redraw bubbles after every rerun
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- CHAT INPUT LOGIC ---
    if user_prompt := st.chat_input("Ask about shipping, taxes, or suppliers..."):
        # Display user message
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing logistics query..."):
                try:
                    response = st.session_state.bot.invoke({"input": user_prompt})
                    ai_text = response["response"]
                    st.markdown(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                except Exception as e:
                    st.error(f"System Error: {str(e)}")

if __name__ == "__main__":
    main()