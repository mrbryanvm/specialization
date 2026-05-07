import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

#load enviroment variables
load_dotenv()

def initialize_import_bot():
    # 1. LLM Setup (Using Llama 3 for speed and cost-efficiency)
    llm = ChatGroq(
        temperature = 0.5,
        model_name = "llama-3.1-8b-instant",
        api_key = os.getenv("GROQ_API_KEY")
    )

    # 2. Advanced System Prompt Template
    # Focus: Constraints, and Tone
    template = """
    You are ClinicBot, a virtual assistant specializing in costumer service for dental clinic
   
    Objective:
    To help patients with clear information, appointment scheduling, 
    and common questions about dental treatments.

    Rules:
    1. Only answer questions related to dental services,, appointments, treatments, 
    approximate prices, and oral care.
    2. If the user asks something outside of this scope, you must politely decline
    and redirect them to dental topics.
    3. DO NOT fabricate complex medical information. If necessary, recommend consulting
    a specialist.
    4. Keep your answer clear, short, and professional.

    Tone:
    -Friendly and professional
    -Clear and direct
    -Focused on helping the patient

    Format:
    -Responses of a maximum of 5 lines
    -Use lists when helpful

    Current conversation:
    {history}
    User: {input}
    ClinicBot:
    """
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    # 3. memory managment
    # Critique: conversationBufferMemory is easy to use but grows indefinitely.
    # For a production app, we'd use ConversationSummaryBufferMemory to save tokens/cost.
    memory = ConversationBufferMemory()

    # 4. Chain Initialization
    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False # Set to True if you want to debug the 'thinking' process
    )

    return conversation

def main():
    clinic_bot = initialize_import_bot()
    print("---Clinic Bot System Active (May 2026)--- \n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("ClinicBot: Goodbye!")
            break

        try:
            response = clinic_bot.invoke({"input": user_input})
            print(f"\nClinicBot: {response['response']}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

