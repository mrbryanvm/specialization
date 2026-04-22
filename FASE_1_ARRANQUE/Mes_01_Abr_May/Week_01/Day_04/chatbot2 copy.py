import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)
def run_chatbot():
    print("===AI Chatbot CLI started===")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    messages = [
        {"role": "system", "content": "You are a helpful and funny progamming assistant."}
    ]
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Happy coding! Bye bitch")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model = "llama-3.1-8b-instant",
                messages = messages
            )
            assistant_response = response.choices[0].message.content
            print(f"\nAssitant: {assistant_response}\n")

            messages.append({"role": "assistant", "content": assistant_response})

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_chatbot()
