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

# Suppress LangChain deprecation warnings for a clean console demo
warnings.filterwarnings("ignore")

load_dotenv()

def initialize_logistics_bot():
    # LLM Setup: Low temperature (0.2) for strict factual accuracy in logistics/customs
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    template = """
    You are 'BoImport Expert', a specialized consultant for Bolivian entrepreneurs importing tech hardware from China.
    Your goal is to provide technical advice on sourcing, shipping (Sea/Air), and Bolivian customs (Aduana Nacional).

    CONSTRAINTS:
    1. ONLY answer questions related to international trade, logistics, hardware specs (SSDs, RAM, CPUs), and Bolivian regulations.
    2. If the user asks about unrelated topics, politely refuse and pivot back to hardware logistics.
    3. NEVER provide financial advice regarding tax evasion; strictly promote legal import paths.
    4. Do not invent exact current tariff percentages if you are unsure; advise checking the official Aduana Nacional tariff book.

    TONE:
    Professional, direct, and highly efficient. Use technical terms but explain them briefly.

    Format:
    - Keep responses under 6 lines unless listing steps.
    - Use Markdown for emphasis and lists.

    Current conversation:
    {history}
    User: {input}
    BoImport Expert:
    """
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    # Critique: ConversationBufferMemory is acceptable for this CLI MVP.
    # For scalable production, migrate to ConversationSummaryBufferMemory to minimize token costs.
    memory = ConversationBufferMemory()

    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False 
    )

    return conversation

def main():
    console = Console()
    logistics_bot = initialize_logistics_bot()
    
    # UI: Professional Welcome Panel
    console.print(Panel.fit(
        "[bold yellow]📦 Welcome to the BoImport Logistics Assistant[/bold yellow]\n[cyan]Type 'exit', 'quit', or 'bye' to terminate.[/cyan]", 
        title="BoImport System (May 2026)", 
        border_style="yellow"
    ))

    while True:
        user_input = console.input("\n[bold green]Importer:[/bold green] ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            console.print("[bold red]BoImport Expert: Session terminated. Safe shipping![/bold red]")
            break

        try:
            # UI: Loading spinner while waiting for LLM response
            with console.status("[bold yellow]Analyzing logistics data...[/bold yellow]", spinner="dots"):
                response = logistics_bot.invoke({"input": user_input})
            
            # UI: Render the LLM's Markdown response inside a styled Panel
            console.print(Panel(
                Markdown(response['response']), 
                title="📦 BoImport Expert", 
                border_style="cyan"
            ))
        except Exception as e:
            console.print(f"[bold red]System Error: {e}[/bold red]")

if __name__ == "__main__":
    main()