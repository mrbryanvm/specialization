# 📦 BoImport AI Logistics Assistant

> A specialized CLI virtual assistant powered by Generative AI to guide Bolivian entrepreneurs in hardware logistics and import procedures from China.

##  Overview
**BoImport Expert** is a highly constrained, specialized AI consultant built for the terminal. Instead of a general-purpose chatbot, this application demonstrates the power of **Contextual Guardrails** and **System Prompt Engineering**. It helps users navigate the complex logistics of importing tech hardware (SSDs, CPUs, RAM) via Air or Sea freight, while strictly adhering to Bolivian customs (Aduana Nacional) regulations.

##  Key Features
- **Strict Domain Constraints:** The AI is programmed to politely refuse off-topic questions, ensuring it remains a logistics tool and not a general conversational bot.
- **Fact-Based Temperature Control:** Uses a low LLM temperature (`0.2`) to heavily reduce hallucinations and prevent the AI from fabricating legal or financial advice (e.g., tax evasion).
- **Conversational Memory:** Implements `ConversationBufferMemory` to maintain the context of the user's questions throughout the active session.
- **Premium CLI Interface:** Replaces standard terminal outputs with `rich`, providing colorful panels, Markdown rendering, and visual loading spinners for a polished user experience.

## 🛠️ Technology Stack
- **Python 3.x**
- **LangChain:** Orchestrates the LLM, Prompt Templates, and Memory.
- **Groq API:** Utilizes the lightning-fast `llama-3.1-8b-instant` model.
- **Rich:** Enhances the command-line interface with beautiful formatting.
- **python-dotenv:** Securely loads API keys from environment variables.

## ⚙️ Installation & Usage

1. **Clone the repository and navigate to the directory:**
   ```bash
   git clone -b feat/week-1 --single-branch <https://github.com/mrbryanvm/specialization.git>
   cd FASE_1_ARRANQUE/Mes_01_Abr_May/Week_01/Day_06
   ```

2. **Install the required dependencies:**
   ```bash
   pip install langchain langchain-groq python-dotenv rich
   ```

3. **Set up the Environment Variables:**
   Create a `.env` file in the same directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_actual_api_key_here
   ```

4. **Run the Assistant:**
   Execute the Python script to start the CLI interface:
   ```bash
   python portfolio_logistics_bot.py
   ```

## 📝 Example Interaction
**Importer:** What are the tariffs for importing RAM from Shenzhen?

**BoImport Expert:** _(Analyzes with loading spinner...)_
> Provides a concise, formatted response focusing strictly on Bolivian Aduana regulations, refusing to invent exact percentages, and maintaining a highly professional tone.
