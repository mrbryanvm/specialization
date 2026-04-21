import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()

# Initialize the Groq/OpenAI client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def run_chatbot():
    print("--- AI Chatbot CLI Started ---")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    # This list will store the entire conversation history (Memory)
    #
    messages = [
        {"role": "system", "content": "You are a helpful and witty Python programming assistant."}
    ]

    while True:
        # Get user input
        user_input = input("You: ")

        # Check for exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Happy coding! Goodbye.")
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        try:
            # Request completion from the AI
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )

            # Extract the content of the response
            assistant_response = response.choices[0].message.content
            print(f"\nAssistant: {assistant_response}\n")

            # CRITICAL: Add assistant's response to history so it has "memory"
            messages.append({"role": "assistant", "content": assistant_response})

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_chatbot()