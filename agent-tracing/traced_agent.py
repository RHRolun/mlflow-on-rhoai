"""
LangChain/LangGraph Agent with MCP integration and MLflow tracing.

This agent demonstrates:
- LangGraph ReAct agent with tool use
- MCP tools via langchain-mcp-adapters
- MLflow autolog tracing (automatic, no manual spans needed)
- MaaS (Model as a Service) OpenAI-compatible endpoint
- Real API integrations: Open-Meteo weather, DuckDuckGo search
"""

import os
import httpx
from typing import Optional, List
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent


@dataclass
class AgentConfig:
    """Configuration for the LangChain agent."""
    model: str = "Llama-4-Scout-17B-16E-W4A16"
    base_url: str = "https://litellm-litemaas.apps.prod.rhoai.rh-aiservices-bu.com/v1"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


def create_llm(config: AgentConfig) -> ChatOpenAI:
    """Create ChatOpenAI instance with MaaS-compatible endpoint."""
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


# =============================================================================
# Local Tool Definitions
# =============================================================================

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """
    Perform basic arithmetic operations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide, sqrt)
        a: First number
        b: Second number (ignored for sqrt)

    Returns:
        The result of the calculation
    """
    import math

    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
        "sqrt": lambda x, _: math.sqrt(x),
        "power": lambda x, y: x ** y,
    }

    if operation not in operations:
        return f"Error: Unknown operation '{operation}'. Use: add, subtract, multiply, divide, sqrt, power"

    result = operations[operation](a, b)
    if operation == "sqrt":
        return f"Square root of {a} = {result}"
    return f"Result of {a} {operation} {b} = {result}"


# =============================================================================
# Open-Meteo Weather API (Free, no API key required)
# https://open-meteo.com/
# =============================================================================

# City coordinates for Open-Meteo API
CITY_COORDINATES = {
    "new york": {"lat": 40.7128, "lon": -74.0060},
    "london": {"lat": 51.5074, "lon": -0.1278},
    "tokyo": {"lat": 35.6762, "lon": 139.6503},
    "sydney": {"lat": -33.8688, "lon": 151.2093},
    "san francisco": {"lat": 37.7749, "lon": -122.4194},
    "paris": {"lat": 48.8566, "lon": 2.3522},
    "berlin": {"lat": 52.5200, "lon": 13.4050},
    "madrid": {"lat": 40.4168, "lon": -3.7038},
    "barcelona": {"lat": 41.3851, "lon": 2.1734},
    "los angeles": {"lat": 34.0522, "lon": -118.2437},
}


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city using Open-Meteo API (free, no API key required).

    Args:
        city: The name of the city to get weather for

    Returns:
        Current weather information for the city
    """
    city_lower = city.lower().strip()

    # Get coordinates
    if city_lower not in CITY_COORDINATES:
        available = ", ".join(CITY_COORDINATES.keys())
        return f"City '{city}' not found. Available cities: {available}"

    coords = CITY_COORDINATES[city_lower]

    try:
        # Call Open-Meteo API (free, no API key required)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit",
        }

        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind_speed = current.get("wind_speed_10m", "N/A")
        weather_code = current.get("weather_code", 0)

        # Weather code to description mapping
        weather_descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
        }
        condition = weather_descriptions.get(weather_code, "Unknown")

        return f"Weather in {city.title()}: {temp}°F, {condition}, {humidity}% humidity, Wind: {wind_speed} km/h"

    except httpx.HTTPError as e:
        return f"Error fetching weather for {city}: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# DuckDuckGo Search (Free, no API key required)
# =============================================================================

# Initialize DuckDuckGo search tool
_duckduckgo_search = None


def get_duckduckgo_search():
    """Get or create DuckDuckGo search instance."""
    global _duckduckgo_search
    if _duckduckgo_search is None:
        _duckduckgo_search = DuckDuckGoSearchRun()
    return _duckduckgo_search


@tool
def search(query: str) -> str:
    """
    Search the web for information using DuckDuckGo.

    Args:
        query: The search query

    Returns:
        Search results for the query
    """
    try:
        ddg = get_duckduckgo_search()
        results = ddg.invoke(query)
        return f"Search results for '{query}':\n{results}"
    except Exception as e:
        return f"Search error: {str(e)}"


# Default local tools list
LOCAL_TOOLS = [calculator, get_weather, search]


# =============================================================================
# Agent Creation Functions
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

When answering questions:
1. Think through what tools might help answer the question
2. Use tools when they would provide useful information
3. Synthesize the results into a clear, helpful response
4. Be concise but thorough in your explanations

You have access to tools for:
- Calculations (add, subtract, multiply, divide, sqrt, power)
- Real-time weather lookups via Open-Meteo API
- Web search via DuckDuckGo
- Travel/flight information via MCP (if enabled)"""


def create_agent_graph(
    config: AgentConfig,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
):
    """
    Create a LangGraph ReAct agent.

    Args:
        config: Agent configuration with LLM settings
        tools: List of tools (uses LOCAL_TOOLS if not provided)
        system_prompt: Custom system prompt (uses DEFAULT_SYSTEM_PROMPT if not provided)

    Returns:
        Compiled LangGraph agent ready for invocation
    """
    llm = create_llm(config)

    if tools is None:
        tools = LOCAL_TOOLS

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Create ReAct agent using langgraph.prebuilt
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    return agent


async def create_agent_with_mcp(
    config: AgentConfig,
    mcp_config: dict,
    include_local_tools: bool = True,
    system_prompt: Optional[str] = None,
):
    """
    Create agent with MCP tools from external servers.

    Args:
        config: Agent configuration with LLM settings
        mcp_config: MCP server configuration dict, e.g.:
            {
                "travel": {
                    "url": "https://mcp.kiwi.com",
                    "transport": "streamable_http"
                }
            }
        include_local_tools: Whether to include local tools (calculator, weather, search)
        system_prompt: Custom system prompt

    Returns:
        Tuple of (compiled agent, MCP client)
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # Initialize MCP client and get tools
    client = MultiServerMCPClient(mcp_config)
    mcp_tools = await client.get_tools()

    # Combine MCP tools with local tools if requested
    all_tools = list(mcp_tools)
    if include_local_tools:
        all_tools.extend(LOCAL_TOOLS)

    # Create the agent with combined toolset
    agent = create_agent_graph(config, all_tools, system_prompt)

    return agent, client


def get_config_from_env() -> AgentConfig:
    """
    Load agent configuration from environment variables.

    Environment variables:
        MAAS_MODEL: Model name (default: Llama-4-Scout-17B-16E-W4A16)
        MAAS_BASE_URL: MaaS endpoint URL
        MAAS_API_KEY: API key for authentication

    Returns:
        AgentConfig populated from environment
    """
    return AgentConfig(
        model=os.environ.get("MAAS_MODEL", "Llama-4-Scout-17B-16E-W4A16"),
        base_url=os.environ.get(
            "MAAS_BASE_URL",
            "https://litellm-litemaas.apps.prod.rhoai.rh-aiservices-bu.com/v1"
        ),
        api_key=os.environ.get("MAAS_API_KEY", ""),
        temperature=0.0,
        max_tokens=4096,
    )


def get_mcp_config_from_env() -> Optional[dict]:
    """
    Get MCP server configuration from environment variables.

    Environment variables:
        MCP_SERVER_ENABLED: Set to 'true' to enable MCP
        MCP_SERVER_URL: URL of the MCP server

    Returns:
        MCP config dict or None if disabled
    """
    if os.environ.get("MCP_SERVER_ENABLED", "false").lower() != "true":
        return None

    mcp_url = os.environ.get("MCP_SERVER_URL", "https://mcp.kiwi.com")
    return {
        "travel": {
            "url": mcp_url,
            "transport": "streamable_http",
        }
    }
