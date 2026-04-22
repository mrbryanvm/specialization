import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = ChatGroq(
    temperature = 0.8,
    model_name = "llama-3.1-8b-instant"
)

template = """
You are a helpful and funny python expert.
Explain the following concept in one short paragraph: {topic}
"""
prompt = PromptTemplate(input_variables = ["topic"], template = template)
chain = prompt | llm
print("Thinking...")
response = chain.invoke({"topic": "Give me pro tips to learn python"})

print("\n ---AI RESPONSE ---")
print(response.content)