import os
import warnings

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

warnings.filterwarnings("ignore")
load_dotenv()


def initialize_medical_bot():
    llm = ChatGroq(
        temperature=0.1,
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
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
        llm=llm, prompt=prompt, memory=memory, verbose=False
    )

    return conversation


def main():
    console = Console()
    medical_bot = initialize_medical_bot()

    console.print(
        Panel.fit(
            "[bold green] Welcome to MedicBot[/bold green]\n[yellow]type 'exit' to terminate.[/yellow]",
            title="MedicBot System",
            border_style="blue",
        )
    )

    while True:
        user_input = console.input("[bold blue]Patient:[/bold blue] ")

        if user_input.lower() in ["exit"]:
            console.print("[bold red]Medicbot: Session Terminated. Bye![/bold red]")
            break

        try:
            with console.status(
                "[bold yellow] Analyzing dental data...[/bold yellow]", spinner="dots"
            ):
                response = medical_bot.invoke({"input": user_input})

            console.print(
                Panel(
                    Markdown(response["response"]),
                    title="MedicBot",
                    border_style="blue",
                )
            )
        except Exception as e:
            console.print(f"[bold red]System Error: {e}[/bold red]")


if __name__ == "__main__":
    main()
