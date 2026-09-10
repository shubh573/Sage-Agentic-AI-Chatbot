# Sage

Sage is an open-source **agentic AI chatbot** built with **Python, FastAPI, LangGraph, LangChain, Google Gemini, Tavily, ChromaDB, and SQLite**.

It supports real-time streaming chat, document uploads, retrieval-augmented generation (RAG), web search, conversation memory, and a simple web UI.

All Popular LLM models like Gemini, OpenAI, Groq and Anthropic at one place


### Virtual environment

pip install virtualenv

python -m virtualenv venv

pip install -r requirements.txt

---

## Environment Variables

Create a `.env` file in the project root directory.

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_google_api_key

TAVILY_API_KEY=your_tavily_api_key

ALPHA_VANTAGE_API_KEY=your_alpha_vintage_api_key

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

LANGSMITH_PROJECT=sage
LANGSMITH_TRACING=true

GOOGLE_MODEL="gemini-3.5-flash"
```

If you do not want to use LangSmith tracing, keep:

```env
LANGSMITH_TRACING=false
```

---

## Run Locally

Start the FastAPI app:

```bash
python app.py
```

The app will be available at:

```text
http://127.0.0.1:8080
```


Run the API server
python -m uvicorn web.app:app --reload

