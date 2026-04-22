import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load API key
load_dotenv()

# 1. Initialize the LLM
llm = ChatGroq(
    temperature=0.7, 
    model_name="llama-3.1-8b-instant"
)

# 2. Initialize the Memory Object
# This object acts as the "brain" and will automatically store the history
memory = ConversationBufferMemory()

# 3. Create the Conversation Chain
# We link the LLM and the Memory together into one powerful object
chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False # Tip: Set this to True later to see how LangChain works behind the scenes!
)

print("🤖 LangChain Chatbot Started! Type 'exit' to end.\n")

# 4. The Conversation Loop
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    # THE MAGIC: We just call .predict()!
    # LangChain automatically:
    # 1. Reads the past memory
    # 2. Adds your new input
    # 3. Sends it all to the AI
    # 4. Extracts the response
    # 5. Saves the response back to the memory
    response = chatbot.predict(input=user_input)
    
    print(f"\nAI: {response}\n")
