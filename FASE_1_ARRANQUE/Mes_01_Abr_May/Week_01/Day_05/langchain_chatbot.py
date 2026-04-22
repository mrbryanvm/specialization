import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate # 1. Add this import

load_dotenv()

llm = ChatGroq(temperature=0.7, model_name="llama-3.1-8b-instant")

# 2. Define the personality
template = """
You are a sarcastic and grumpy (but highly skilled) senior Python developer. 
You help the user, but you complain about how easy their questions are.

Current conversation:
{history}
Human: {input}
AI:"""

custom_prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# 3. Initialize memory
memory = ConversationBufferMemory()

# 4. Create the chain with the CUSTOM PROMPT
chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=custom_prompt, # This connects the personality to the memory
    verbose=False 
)

print("--- Grumpy Dev Chatbot (Type 'exit' to quit) ---")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
        
    response = chatbot.predict(input=user_input)
    print(f"Grumpy Dev: {response}")
    print("-----------------------------------------------\n")