import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = ChatGroq(
    temperature = 0.9,
    model_name = "llama-3.1-8b-instant"
)
template = """
You are a sarcastic and grumpy (but highly skilled) senior Python developer.
You help the user, but you complain a lot about how easy their questions are.
Current conversation:
{history}
Human: {input}
AI:
"""
custom_prompt = PromptTemplate(input_variables = ["history", "input"], template = template)
memory = ConversationBufferMemory()
chatbot = ConversationChain(
    llm = llm,
    memory = memory,
    prompt = custom_prompt,
    verbose = True
)

print("--- Grumpy Dev Chatbot (Type 'exit' to quit) ---")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    response = chatbot.predict(input = user_input)
    print(f"Grumpy Dev: {response}")
    print("---------------------------------------\n")