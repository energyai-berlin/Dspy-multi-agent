"""
DSPy Toy Example: Agent with Two Subagents using ReAct

This example demonstrates:
- Two subagents (MathAgent and TextAgent)
- Each subagent has two tools
- Subagents use dspy.ReAct with proper Signatures
- Main agent wraps subagents and delegates tasks
"""


import dspy
from typing import List
import json
from datetime import datetime
import pytz
import requests
import time
from functools import wraps



def print_react_trajectory(prediction: dspy.Prediction):
    """Print ReAct trajectory in 'thinking → action → results' format"""
    if not hasattr(prediction, 'trajectory') or not prediction.trajectory:
        return

    trajectory = prediction.trajectory

    max_iter = 0
    while f'thought_{max_iter}' in trajectory:
        max_iter += 1

    # Print each iteration
    for i in range(max_iter):
        thought = trajectory.get(f'thought_{i}', '')
        tool_name = trajectory.get(f'tool_name_{i}', '')
        tool_args = trajectory.get(f'tool_args_{i}', {})
        observation = trajectory.get(f'observation_{i}', '')

        print(f"\n   💭 Thinking: {thought}")
        print(f"   🔧 Action: {tool_name}({tool_args})")
        print(f"   📊 Result: {observation}")


def trace_agent(agent_name: str):
    """Decorator to trace agent execution with visual hierarchy"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Show agent started
            print(f"\n🤖 [{agent_name}] STARTED")
            print(f"   📥 Input: {kwargs}")

            start_time = time.time()

            # Execute the agent
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Print trajectory in thinking → action → results format
                if isinstance(result, dspy.Prediction):
                    print_react_trajectory(result)

                    # Show final answer
                    print("\n   ✨ Final Answer:")

                    for key, value in result.items():
                        if key not in ['trajectory', 'reasoning']:
                            print(f"      {key}: {value}")

                print(f"\n   ⏱️  Completed in {elapsed:.2f}s")
                print(f"✅ [{agent_name}] FINISHED\n")

                return result
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"   ❌ Error: {str(e)}")
                print(f"   ⏱️  Failed after {elapsed:.2f}s")
                print(f"❌ [{agent_name}] FAILED\n")
                raise
        return wrapper
    return decorator

# ============================================================================
# SUBAGENT 1: MathAgent with two tools
# ============================================================================

# Tool 1: Add numbers
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the sum."""
    return a + b


# Tool 2: Multiply numbers
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together and return the product."""
    return a * b


# Math tools collection
math_tools = [add_numbers, multiply_numbers]


class MathAgentSignature(dspy.Signature):
    """Signature for mathematical operations agent"""
    math_query: str = dspy.InputField(desc="A mathematical question or operation request")
    math_result: str = dspy.OutputField(desc="The result of the mathematical operation")


class MathAgent(dspy.Module):
    """Agent that handles mathematical operations using ReAct."""

    def __init__(self):
        super().__init__()
        self.react_program = dspy.ReAct(
            signature=MathAgentSignature,
            tools=math_tools,
            max_iters=3
        )

    @trace_agent("MathAgent")
    def forward(self, math_query: str) -> dspy.Prediction:
        """Process mathematical queries and return results"""
        return self.react_program(math_query=math_query)


# ============================================================================
# SUBAGENT 2: TextAgent with two tools
# ============================================================================

# Tool 1: Count words
def count_words(text: str) -> int:
    """Count the number of words in the given text."""
    return len(text.split())


# Tool 2: Reverse text
def reverse_text(text: str) -> str:
    """Reverse the given text and return it backwards."""
    return text[::-1]


# Text tools collection
text_tools = [count_words, reverse_text]


class TextAgentSignature(dspy.Signature):
    """Signature for text processing operations agent"""
    text_query: str = dspy.InputField(desc="A text processing question or operation request")
    text_result: str = dspy.OutputField(desc="The result of the text processing operation")


class TextAgent(dspy.Module):
    """Agent that handles text operations using ReAct."""

    def __init__(self):
        super().__init__()
        self.react_program = dspy.ReAct(
            signature=TextAgentSignature,
            tools=text_tools,
            max_iters=3
        )

    @trace_agent("TextAgent")
    def forward(self, text_query: str) -> dspy.Prediction:
        """Process text queries and return results"""
        return self.react_program(text_query=text_query)


# ============================================================================
# SUBAGENT 3: TimeAgent with two tools
# ============================================================================

# Tool 1: Get USA time
def get_usa_time() -> str:
    """Get the current time in USA (Eastern Time)."""
    usa_tz = pytz.timezone('America/New_York')
    usa_time = datetime.now(usa_tz)
    return usa_time.strftime("%Y-%m-%d %H:%M:%S %Z")


# Tool 2: Get China time
def get_china_time() -> str:
    """Get the current time in China (Beijing Time)."""
    china_tz = pytz.timezone('Asia/Shanghai')
    china_time = datetime.now(china_tz)
    return china_time.strftime("%Y-%m-%d %H:%M:%S %Z")


# Time tools collection
time_tools = [get_usa_time, get_china_time]


class TimeAgentSignature(dspy.Signature):
    """Find the timezone or location in the input query and return the corresponding time."""
    time_query: str = dspy.InputField(desc="A time-related question or timezone request")
    time_result: str = dspy.OutputField(desc="The current time in the requested timezone")


class TimeAgent(dspy.Module):
    """Agent that handles time operations using ReAct."""

    def __init__(self):
        super().__init__()
        self.react_program = dspy.ReAct(
            signature=TimeAgentSignature,
            tools=time_tools,
            max_iters=3
        )

    @trace_agent("TimeAgent")
    def forward(self, time_query: str) -> dspy.Prediction:
        """Process time queries and return results"""
        return self.react_program(time_query=time_query)


# ============================================================================
# SUBAGENT 4: WeatherAgent with two tools
# ============================================================================

# Tool 1: Get weather by city name
def get_weather_by_city(city_name: str) -> str:
    """Get current weather information for a given city using Open-Meteo API."""
    try:
        # First, get coordinates for the city using geocoding
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url, timeout=10)
        geo_data = geo_response.json()

        if not geo_data.get('results'):
            return f"City '{city_name}' not found"

        location = geo_data['results'][0]
        lat = location['latitude']
        lon = location['longitude']
        city = location['name']
        country = location.get('country', 'Unknown')

        # Get weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        weather_response = requests.get(weather_url, timeout=10)
        weather_data = weather_response.json()

        current = weather_data['current']
        temp = current['temperature_2m']
        humidity = current['relative_humidity_2m']
        wind_speed = current['wind_speed_10m']

        return f"Weather in {city}, {country}: Temperature: {temp}°C, Humidity: {humidity}%, Wind Speed: {wind_speed} km/h"

    except Exception as e:
        return f"Error fetching weather: {str(e)}"


# Tool 2: Get temperature comparison between two cities
def compare_city_temperatures(city1: str, city2: str) -> str:
    """Compare temperatures between two cities."""
    try:
        temps = {}
        for city in [city1, city2]:
            # Get coordinates
            geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            geo_response = requests.get(geocoding_url, timeout=10)
            geo_data = geo_response.json()

            if not geo_data.get('results'):
                return f"City '{city}' not found"

            location = geo_data['results'][0]
            lat = location['latitude']
            lon = location['longitude']

            # Get weather
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
            weather_response = requests.get(weather_url, timeout=10)
            weather_data = weather_response.json()

            temps[city] = weather_data['current']['temperature_2m']

        diff = abs(temps[city1] - temps[city2])
        warmer = city1 if temps[city1] > temps[city2] else city2

        return f"Temperature comparison: {city1}: {temps[city1]}°C, {city2}: {temps[city2]}°C. {warmer} is warmer by {diff}°C"

    except Exception as e:
        return f"Error comparing temperatures: {str(e)}"


# Weather tools collection
weather_tools = [get_weather_by_city, compare_city_temperatures]


class WeatherAgentSignature(dspy.Signature):
    """Find weather and temperature information and structure into the requested output form."""
    weather_query: str = dspy.InputField(desc="A weather-related question about cities or temperature comparison")
    weather_result: str = dspy.OutputField(desc="Weather information or temperature comparison results")


class WeatherAgent(dspy.Module):
    """Agent that handles weather operations using ReAct."""

    def __init__(self):
        super().__init__()
        self.react_program = dspy.ReAct(
            signature=WeatherAgentSignature,
            tools=weather_tools,
            max_iters=3
        )

    @trace_agent("WeatherAgent")
    def forward(self, weather_query: str) -> dspy.Prediction:
        """Process weather queries and return results"""
        return self.react_program(weather_query=weather_query)


# ============================================================================
# WRAP FUNCTIONS: Callable wrappers for subagents
# ============================================================================

def math_calculator(math_query: str) -> str:
    """
    Math Calculator Agent: Performs mathematical operations
    - **Addition**: Adds two numbers together
    - **Multiplication**: Multiplies two numbers together
    - **Smart Operation Selection**: ReAct decides which tool to use
    - Output key: "math_result"
    """
    math_agent = MathAgent()
    prediction = math_agent(math_query=math_query)
    return prediction.math_result


def text_processor(text_query: str) -> str:
    """
    Text Processor Agent: Performs text operations
    - **Word Counter**: Counts words in text
    - **Text Reverser**: Reverses text backwards
    - **Smart Operation Selection**: ReAct decides which tool to use
    - Output key: "text_result"
    """
    text_agent = TextAgent()
    prediction = text_agent(text_query=text_query)
    return prediction.text_result


def time_checker(time_query: str) -> str:
    """
    Time Checker Agent: Provides current time information
    - **USA Time**: Returns current time in USA (Eastern Time)
    - **China Time**: Returns current time in China (Beijing Time)
    - **Smart Timezone Selection**: ReAct decides which tool to use
    - Output key: "time_result"
    """
    time_agent = TimeAgent()
    prediction = time_agent(time_query=time_query)
    return prediction.time_result


def weather_checker(weather_query: str) -> str:
    """
    Weather Checker Agent: Provides real-time weather information
    - **City Weather**: Gets current weather for any city worldwide
    - **Temperature Comparison**: Compares temperatures between two cities
    - **Smart Tool Selection**: ReAct decides which weather tool to use
    - Output key: "weather_result"
    """
    weather_agent = WeatherAgent()
    prediction = weather_agent(weather_query=weather_query)
    return prediction.weather_result


# ============================================================================
# MAIN AGENT: Wraps subagents
# ============================================================================

class MainAgentSignature(dspy.Signature):
    """You are an intelligent coordinator that breaks down complex queries into sub-tasks.

    Your responsibilities:
    1. Analyze the user's query and identify distinct tasks (math, text, time, weather)
    2. Break multi-part queries into separate sub-queries for each specialist agent
    3. Call the appropriate agent tools in sequence to gather all needed information
    4. Synthesize results from multiple agents into a coherent final answer

    Available agents and their capabilities:
    - math_calculator: Mathematical operations (addition, multiplication)
    - text_processor: Text operations (word count, text reversal)
    - time_checker: Time queries (USA time, China time)
    - weather_checker: Weather queries (city weather, temperature comparisons)

    For multi-part queries like "What's 5+3 and weather in Cairo?", call math_calculator first,
    then weather_checker, and combine both results in your final answer.
    """
    user_query: str = dspy.InputField(desc="User's potentially multi-part query requiring one or more specialist agents")
    final_answer: str = dspy.OutputField(desc="Complete answer combining results from all relevant agents")


class MainAgent(dspy.Module):
    """Main coordinator agent that intelligently delegates to specialist subagents.

    This agent can:
    - Handle single queries: "What's the time in USA?" → routes to time_checker
    - Handle multi-part queries: "What's 5+3 and weather in Cairo?" → routes to both math_calculator and weather_checker
    - Coordinate multiple agent calls in sequence
    - Synthesize results from multiple agents into coherent answers
    """

    def __init__(self):
        super().__init__()
        # Configure ReAct with all specialist agents as tools
        self.root_program = dspy.ReAct(
            signature=MainAgentSignature,
            tools=[math_calculator, text_processor, time_checker, weather_checker],
            max_iters=5  # Allow multiple iterations for multi-part queries
        )

    def forward(self, user_query: str):
        results = self.root_program(user_query=user_query)
        return results
    



dspy.configure(lm=dspy.LM('ollama_chat/qwen3:1.7b', api_base='http://localhost:11434', api_key=''))

root_agent = MainAgent()
if __name__ == "__main__":

    test_query = "Can you tell me the weather in berlin and the current time in China?"
    result = root_agent(user_query=test_query)
    print(result.final_answer)