import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)
print("Thinking...")
messages = [
    {"role": "system", "content": "You are a Python programming assistant. Be brief and ironic"},
    {"role": "user", "content": "Explain how to learn to dance urban style"}
]
try:
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = messages
    )
    print("\n===Response from LLM===")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")
