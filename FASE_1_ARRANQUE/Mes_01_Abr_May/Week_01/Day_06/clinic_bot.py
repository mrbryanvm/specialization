import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown 

warnings.filterwarnings("ignore")
load_dotenv()

def initialize_medical_bot():
    llm = ChatGroq(
        temperature = 0.1,
        model_name = "llama-3.1-8b-instant",
        api_key = os.getenv("GROQ_API_KEY")
    )

    template = """
    You are 'MedicBot', a specialized dental assistant.
    Your goal is to provide dental care advices.

    CONSTRAINTS:
    1. Only answer questions related to dental care.
    2. If the user asks about unrelated topics, politely refuse and pivot to dental topics.

    TONE:
    - Empathetic but direct.
    - Funny and witty

    FORMAT:
    - Keep responses under 6 lines unless listing steps.
    - Use Markdown for emphasis and lists.

    Current conversation:
    {history}
    User: {input}
    Medicbot:
    """
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    memory = ConversationBufferMemory()

    conversation = ConversationChain(
        llm = llm,
        prompt = prompt,
        memory = memory,
        verbose = False
    )

def main():
    console = Console()
    medical_bot = initialize_medical_bot()

    console.print(Panel.fit(
        "[bold green] Welcome to MedicBot[/bold green]\n[yellow]type 'exit' to terminate.[/yellow]",
        title = "MedicBot System",
        border_style = "blue"
    ))

    while True:
        user_input