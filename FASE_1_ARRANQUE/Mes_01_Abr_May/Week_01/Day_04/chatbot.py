import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the client
# We use os.getenv to keep the API Key secret
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

print("Thinking...")

# Define the conversation context
# Day 4 Objective: Use a list to eventually handle history
messages = [
    {"role": "system", "content": "You are a Python programming assistant. Be brief and use humor."},
    {"role": "user", "content": "Explain what a variable is in programming."}
]

# Make the request (Chat Completion)
try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    # Print the AI's response
    print("\n--- Response from the AI ---")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")