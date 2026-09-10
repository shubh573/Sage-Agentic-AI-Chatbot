import math
import os

import requests
from requests.exceptions import RequestException
from contextvars import ContextVar

from dotenv import load_dotenv


load_dotenv()

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from database import save_memory, search_memory
from rag import retrieve_from_rag


# ---------------------------------------------------------
# Request/thread context
# ---------------------------------------------------------

CURRENT_THREAD_ID = ContextVar(
    "current_thread_id",
    default="default"
)


def set_current_thread_id(thread_id: str):
    CURRENT_THREAD_ID.set(thread_id)


def get_current_thread_id() -> str:
    return CURRENT_THREAD_ID.get()


# ---------------------------------------------------------
# Web search
# ---------------------------------------------------------

web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


# ---------------------------------------------------------
# Calculator
# ---------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.

    Input should be a valid math expression.

    Examples:
    2 + 2
    math.sqrt(16)
    10 * 5
    """

    try:

        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed
        )

        return str(result)

    except Exception as e:

        return f"Calculator Error: {str(e)}"


# ---------------------------------------------------------
# RAG
# ---------------------------------------------------------

@tool
def search_uploaded_document(query: str) -> str:
    """
    Search uploaded documents for relevant information.

    Use this when the user asks about uploaded PDFs,
    DOCX, TXT, notes, files, or documents.
    """

    thread_id = get_current_thread_id()

    return retrieve_from_rag(
        query=query,
        thread_id=thread_id
    )


# ---------------------------------------------------------
# Memory
# ---------------------------------------------------------

@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact
    into long-term memory.
    """

    thread_id = get_current_thread_id()

    return save_memory(
        thread_id=thread_id,
        memory=memory
    )


@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories.
    """

    thread_id = get_current_thread_id()

    return search_memory(
        thread_id=thread_id,
        query=query
    )

# ============================================================
# STOCK PRICE
# ============================================================

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock quote for a given stock symbol.

    Examples:
        AAPL
        TSLA
        MSFT
        NVDA
        AMZN

    Use this tool when the user asks for:
    - current stock price
    - latest stock price
    - stock quote
    - share price
    """

    symbol = symbol.strip().upper()

    if not symbol:
        return {
            "success": False,
            "error": "Stock symbol is required."
        }

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "success": False,
            "error": "ALPHA_VANTAGE_API_KEY is not configured."
        }

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        # Alpha Vantage may return an API limit/message
        if "Note" in data:
            return {
                "success": False,
                "error": data["Note"]
            }

        if "Information" in data:
            return {
                "success": False,
                "error": data["Information"]
            }

        quote = data.get("Global Quote", {})

        if not quote:
            return {
                "success": False,
                "symbol": symbol,
                "error": (
                    f"No stock data found for symbol '{symbol}'. "
                    "Check that the ticker is valid."
                )
            }

        return {
            "success": True,
            "symbol": quote.get("01. symbol", symbol),
            "open": quote.get("02. open"),
            "high": quote.get("03. high"),
            "low": quote.get("04. low"),
            "price": quote.get("05. price"),
            "volume": quote.get("06. volume"),
            "latest_trading_day": quote.get(
                "07. latest trading day"
            ),
            "previous_close": quote.get(
                "08. previous close"
            ),
            "change": quote.get(
                "09. change"
            ),
            "change_percent": quote.get(
                "10. change percent"
            )
        }

    except RequestException as e:
        return {
            "success": False,
            "symbol": symbol,
            "error": f"Stock API request failed: {str(e)}"
        }

    except Exception as e:
        return {
            "success": False,
            "symbol": symbol,
            "error": f"Stock lookup failed: {str(e)}"
        }



# ============================================================
# WEATHER
# ============================================================

@tool
def get_weather(location: str) -> str:
    """ 
    Get the current weather for a given location. 
    
    The location can be: 
    - city name 
    - city and country 
    - city and state 
    - postal code 
    
    Examples: 
    - London 
    - London, UK 
    - New York, USA 
    - Mumbai, India 
    - Toronto, Canada 
    
    Use this tool whenever the user asks for current weather,
     temperature, rain, humidity, wind, or weather conditions
      for a location. 
    """

    try:
        location = location.strip()

        if not location:
            return "Please provide a location."

        # ----------------------------------------------------
        # STEP 1: Geocode location
        # ----------------------------------------------------

        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

        geocoding_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        }

        geocoding_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10
        )

        geocoding_response.raise_for_status()

        geocoding_data = geocoding_response.json()

        results = geocoding_data.get("results", [])

        if not results:
            return f"I couldn't find a location matching '{location}'."

        place = results[0]

        latitude = place["latitude"]
        longitude = place["longitude"]

        city_name = place.get("name", location)
        country = place.get("country", "")
        admin1 = place.get("admin1", "")
        timezone = place.get("timezone", "auto")

        # ----------------------------------------------------
        # STEP 2: Get current weather
        # ----------------------------------------------------

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "precipitation,"
                "rain,"
                "weather_code,"
                "cloud_cover,"
                "wind_speed_10m,"
                "wind_direction_10m,"
                "wind_gusts_10m"
            ),
            "timezone": timezone,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data.get("current")

        if not current:
            return f"Weather data is currently unavailable for {location}."

        # ----------------------------------------------------
        # Weather code description
        # ----------------------------------------------------

        weather_code = current.get("weather_code")
        weather_description = get_weather_description(weather_code)

        # ----------------------------------------------------
        # Format result
        # ----------------------------------------------------

        location_parts = [
            city_name,
            admin1,
            country
        ]

        formatted_location = ", ".join(
            part for part in location_parts if part
        )

        temperature = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        precipitation = current.get("precipitation")
        rain = current.get("rain")
        cloud_cover = current.get("cloud_cover")
        wind_speed = current.get("wind_speed_10m")
        wind_direction = current.get("wind_direction_10m")
        wind_gusts = current.get("wind_gusts_10m")
        observation_time = current.get("time")

        return (
            f"Current weather for {formatted_location}:\n"
            f"Condition: {weather_description}\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%\n"
            f"Precipitation: {precipitation} mm\n"
            f"Rain: {rain} mm\n"
            f"Cloud cover: {cloud_cover}%\n"
            f"Wind: {wind_speed} km/h\n"
            f"Wind direction: {wind_direction}°\n"
            f"Wind gusts: {wind_gusts} km/h\n"
            f"Local observation time: {observation_time}\n"
            f"Timezone: {timezone}\n"
            f"Data source: Open-Meteo"
        )


    except requests.RequestException as e:
        return f"Weather service request failed: {str(e)}"

    except Exception as e:
        return f"Weather lookup failed: {str(e)}"


def get_weather_description(weather_code):
    """
    Convert Open-Meteo WMO weather codes into human-readable text.
    """

    descriptions = {
        0: "Clear sky",

        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",

        45: "Fog",
        48: "Depositing rime fog",

        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",

        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",

        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",

        66: "Light freezing rain",
        67: "Heavy freezing rain",

        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",

        77: "Snow grains",

        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",

        85: "Slight snow showers",
        86: "Heavy snow showers",

        95: "Thunderstorm",

        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    return descriptions.get(
        weather_code,
        f"Unknown weather condition (code {weather_code})"
    )


@tool
def get_weather_forecast(location: str, days: int = 5) -> str:
    """
    Get the weather forecast for a location.

    Args:
        location: City, country, state, or postal code.
        days: Number of forecast days. Maximum 7.

    Examples:
        get_weather_forecast("Mumbai", 5)
        get_weather_forecast("London, UK", 7)
        get_weather_forecast("New York, USA", 3)

    Use this tool when the user asks about:
    - tomorrow's weather
    - weather forecast
    - weather for the next few days
    - whether it will rain
    - future temperature
    - future weather conditions
    """

    try:
        # ----------------------------------------------------
        # Validate days
        # ----------------------------------------------------

        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 5

        days = max(1, min(days, 7))

        location = location.strip()

        if not location:
            return "Please provide a location."

        # ----------------------------------------------------
        # STEP 1: Geocode location
        # ----------------------------------------------------

        geocoding_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        geocoding_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        }

        response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10
        )

        response.raise_for_status()

        geocoding_data = response.json()

        results = geocoding_data.get("results", [])

        if not results:
            return (
                f"I couldn't find a location matching "
                f"'{location}'."
            )

        place = results[0]

        latitude = place["latitude"]
        longitude = place["longitude"]

        city_name = place.get("name", location)
        country = place.get("country", "")
        admin1 = place.get("admin1", "")
        timezone = place.get("timezone", "auto")

        # ----------------------------------------------------
        # STEP 2: Weather forecast
        # ----------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,

            "daily": (
                "weather_code,"
                "temperature_2m_max,"
                "temperature_2m_min,"
                "apparent_temperature_max,"
                "apparent_temperature_min,"
                "precipitation_sum,"
                "rain_sum,"
                "precipitation_probability_max,"
                "wind_speed_10m_max,"
                "wind_gusts_10m_max"
            ),

            "forecast_days": days,

            "timezone": timezone,

            "temperature_unit": "celsius",

            "wind_speed_unit": "kmh",

            "precipitation_unit": "mm",
        }

        response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        response.raise_for_status()

        weather_data = response.json()

        daily = weather_data.get("daily")

        if not daily:
            return (
                f"Forecast data is unavailable "
                f"for {location}."
            )

        # ----------------------------------------------------
        # Extract daily data
        # ----------------------------------------------------

        dates = daily.get("time", [])

        weather_codes = daily.get(
            "weather_code",
            []
        )

        max_temperatures = daily.get(
            "temperature_2m_max",
            []
        )

        min_temperatures = daily.get(
            "temperature_2m_min",
            []
        )

        feels_like_max = daily.get(
            "apparent_temperature_max",
            []
        )

        feels_like_min = daily.get(
            "apparent_temperature_min",
            []
        )

        precipitation = daily.get(
            "precipitation_sum",
            []
        )

        rain = daily.get(
            "rain_sum",
            []
        )

        precipitation_probability = daily.get(
            "precipitation_probability_max",
            []
        )

        wind_speed = daily.get(
            "wind_speed_10m_max",
            []
        )

        wind_gusts = daily.get(
            "wind_gusts_10m_max",
            []
        )

        # ----------------------------------------------------
        # Format location
        # ----------------------------------------------------

        location_parts = [
            city_name,
            admin1,
            country
        ]

        formatted_location = ", ".join(
            part
            for part in location_parts
            if part
        )

        # ----------------------------------------------------
        # Format forecast
        # ----------------------------------------------------

        forecast_lines = []

        for i, date in enumerate(dates):

            weather_code = (
                weather_codes[i]
                if i < len(weather_codes)
                else None
            )

            description = get_weather_description(
                weather_code
            )

            max_temp = (
                max_temperatures[i]
                if i < len(max_temperatures)
                else "N/A"
            )

            min_temp = (
                min_temperatures[i]
                if i < len(min_temperatures)
                else "N/A"
            )

            feels_max = (
                feels_like_max[i]
                if i < len(feels_like_max)
                else "N/A"
            )

            feels_min = (
                feels_like_min[i]
                if i < len(feels_like_min)
                else "N/A"
            )

            precipitation_amount = (
                precipitation[i]
                if i < len(precipitation)
                else "N/A"
            )

            rain_amount = (
                rain[i]
                if i < len(rain)
                else "N/A"
            )

            rain_probability = (
                precipitation_probability[i]
                if i < len(precipitation_probability)
                else "N/A"
            )

            max_wind = (
                wind_speed[i]
                if i < len(wind_speed)
                else "N/A"
            )

            max_gust = (
                wind_gusts[i]
                if i < len(wind_gusts)
                else "N/A"
            )

            forecast_lines.append(
                f"{date}:\n"
                f"  Condition: {description}\n"
                f"  Temperature: {min_temp}°C - {max_temp}°C\n"
                f"  Feels like: {feels_min}°C - {feels_max}°C\n"
                f"  Precipitation: {precipitation_amount} mm\n"
                f"  Rain: {rain_amount} mm\n"
                f"  Precipitation probability: {rain_probability}%\n"
                f"  Maximum wind: {max_wind} km/h\n"
                f"  Maximum wind gusts: {max_gust} km/h"
            )

        return (
            f"Weather forecast for {formatted_location}\n"
            f"Forecast period: {days} days\n"
            f"Timezone: {timezone}\n\n"
            + "\n\n".join(forecast_lines)
            + "\n\nData source: Open-Meteo"
        )

    except RequestException as e:
        return (
            f"Weather forecast service request failed: "
            f"{str(e)}"
        )

    except Exception as e:
        return (
            f"Weather forecast lookup failed: "
            f"{str(e)}"
        )


# ---------------------------------------------------------
# Tool list
# ---------------------------------------------------------

tools = [
    calculator,
    search_uploaded_document,
    remember_this,
    recall_memory,
    web_search,
    get_stock_price,
    get_weather,
    get_weather_forecast
]
