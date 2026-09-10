import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from tools import tools


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


Path("data").mkdir(exist_ok=True)


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL",
    "google:gemini-3.5-flash"
)


# Friendly model configuration.
#
# The key is the value sent by the frontend.
# The provider/model information is kept in one place.
#s
# You can add more models later without changing the
# LangGraph workflow.
# ---------------------------------------------------------

MODEL_CONFIG = {

    # =====================================================
    # Google Gemini
    # =====================================================

    "google:gemini-3.8-flash": {
        "provider": "google",
        "model": "gemini-3.8-flash",
        "display_name": "Gemini 3.8 Flash",
    },

    "google:gemini-3.5-flash": {
        "provider": "google",
        "model": "gemini-3.5-flash",
        "display_name": "Gemini 3.5 Flash",
    },

    "google:gemini-3.5-flash-lite": {
        "provider": "google",
        "model": "gemini-3.5-flash-lite",
        "display_name": "Gemini 3.5 Flash Lite",
    },


    # =====================================================
    # OpenAI / ChatGPT
    # =====================================================

    "openai:gpt-5.4": {
        "provider": "openai",
        "model": "gpt-5.4",
        "display_name": "GPT-5.4",
    },

    "openai:gpt-5-mini": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "display_name": "GPT-5 Mini",
    },

    "openai:gpt-5-nano": {
        "provider": "openai",
        "model": "gpt-5-nano",
        "display_name": "GPT-5 Nano",
    },


    # =====================================================
    # Anthropic Claude
    # =====================================================

    "anthropic:claude-sonnet-4-5": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "display_name": "Claude Sonnet",
    },

    "anthropic:claude-opus-4-1": {
        "provider": "anthropic",
        "model": "claude-opus-4-1",
        "display_name": "Claude Opus",
    },


    # =====================================================
    # Groq
    # =====================================================

    "groq:llama-3.3-70b-versatile": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "display_name": "Llama 3.3 70B",
    },

    "groq:llama-4-scout-17b-16e-instruct": {
        "provider": "groq",
        "model": "llama-4-scout-17b-16e-instruct",
        "display_name": "Llama 4 Scout",
    },
}


ALLOWED_MODELS = set(MODEL_CONFIG.keys())


# ---------------------------------------------------------
# System prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named Sage.

You can:

1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Get stock price for given Symbol using the Stock Price tool.
6. Get weather info for a location using the weather tool.
7. Remember important user information using the memory tool.
8. Recall memory when useful.
9. Use calculator for math.

Rules:

- If the user asks about latest news, current events, recent updates,
  today's information, current price, current weather, current stock
  price, or other information that can change, use web search.

- If the user asks about an uploaded document, use
  search_uploaded_document.

- If the user asks you to remember something, use remember_this.

- If the user asks about previous preferences or saved facts,
  use recall_memory.

- Use calculator for math questions.

- If the user asks for a current/latest stock price or stock quote,
  use get_stock_price.

- Always use get_stock_price for current stock prices instead of guessing.

- If the user asks about stock news, company news, earnings, analyst opinions,
  or other information that requires current web information, use web_search.

- Do not claim that a stock price is real-time unless the tool/API actually
  provides real-time data. Clearly identify the latest trading day when available.

- If the user asks about current weather, temperature, rain, humidity, 
  wind, or weather conditions right now, use get_weather.

- If the user asks about tomorrow's weather, future weather, weather forecast, 
  rain prediction, or weather for the next few days, use get_weather_forecast. 

- Do not guess current or future weather. Always use the appropriate weather tool.

- When using web search, summarize the results clearly.

- Be clear, helpful, and concise.
"""


# ---------------------------------------------------------
# Model validation
# ---------------------------------------------------------

def normalize_model_name(model_name: str | None) -> str:
    """
    Validate the model selected by the frontend.

    If the model is missing or unsupported, use DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


# ---------------------------------------------------------
# Model factory
# ---------------------------------------------------------

def create_llm(model_name: str):
    """
    Create the appropriate LangChain chat model.

    Supported providers:

    - Google Gemini
    - OpenAI / ChatGPT
    - Anthropic Claude
    - Groq
    """

    config = MODEL_CONFIG[model_name]

    provider = config["provider"]
    actual_model = config["model"]

    if provider == "google":

        return ChatGoogleGenerativeAI(
            model=actual_model,
            temperature=0.3,
            streaming=True,
        )

    if provider == "openai":

        return ChatOpenAI(
            model=actual_model,
            temperature=0.3,
            streaming=True,
        )

    if provider == "anthropic":

        return ChatAnthropic(
            model=actual_model,
            temperature=0.3,
            streaming=True,
        )

    if provider == "groq":

        return ChatGroq(
            model=actual_model,
            temperature=0.3,
            streaming=True,
        )

    raise ValueError(
        f"Unsupported provider: {provider}"
    )


# ---------------------------------------------------------
# Build LangGraph agent
# ---------------------------------------------------------

# Create one shared SQLite connection.
#
# This avoids creating a separate connection every time
# get_agent() creates a model-specific agent.
# ---------------------------------------------------------

CHECKPOINT_DB = sqlite3.connect(
    "data/langgraph_checkpoints.sqlite",
    check_same_thread=False
)

CHECKPOINTER = SqliteSaver(CHECKPOINT_DB)


def build_agent(model_name: str):
    """
    Build one LangGraph agent for the selected model.
    """

    selected_model = normalize_model_name(model_name)

    llm = create_llm(selected_model)

    # All supported models use LangChain's standard
    # bind_tools interface.
    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):

        conversation_messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + state["messages"]

        response = llm_with_tools.invoke(
            conversation_messages
        )

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node(
        "chatbot",
        chatbot_node
    )

    workflow.add_node(
        "tools",
        tool_node
    )

    workflow.add_edge(
        START,
        "chatbot"
    )

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition
    )

    workflow.add_edge(
        "tools",
        "chatbot"
    )

    return workflow.compile(
        checkpointer=CHECKPOINTER
    )


# ---------------------------------------------------------
# Agent cache
# ---------------------------------------------------------

_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return a cached LangGraph agent.

    Each model gets its own agent instance.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:

        _AGENT_CACHE[selected_model] = build_agent(
            selected_model
        )

    return _AGENT_CACHE[selected_model]


# ---------------------------------------------------------
# Models API helper
# ---------------------------------------------------------

def get_available_models():
    """
    Return models in a frontend-friendly format.
    """

    return [
        {
            "id": model_id,
            "provider": config["provider"],
            "model": config["model"],
            "name": config["display_name"],
        }
        for model_id, config in MODEL_CONFIG.items()
    ]
