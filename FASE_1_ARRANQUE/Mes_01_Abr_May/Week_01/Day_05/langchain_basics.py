import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load the secret key
load_dotenv()

# 1. Initialize the Model
llm = ChatGroq(
    temperature=0.7, # 0.0 is robotic, 1.0 is highly creative
    model_name="llama-3.1-8b-instant"
)

# 2. Create a Prompt Template
# Instead of hardcoding the user's question, we leave a {topic} placeholder
template = """
You are a helpful and funny Python expert.
Explain the following concept in one short paragraph: {topic}
"""
prompt = PromptTemplate(input_variables=["topic"], template=template)

# 3. Create the Chain
# We use the "|" symbol to chain the prompt to the model (This is called LCEL syntax)
chain = prompt | llm

# 4. Run the chain!
print("Thinking...")
response = chain.invoke({"topic": "What is a 'for loop'?"})

print("\n--- AI Response ---")
print(response.content)
