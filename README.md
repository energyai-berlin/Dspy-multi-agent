# DSPy Toy Example: Multi-Agent ReAct System

A demonstration of building intelligent agent systems using [DSPy](https://github.com/stanfordnlp/dspy) with the ReAct pattern.

## Overview

This project showcases a hierarchical agent architecture where a main coordinator delegates tasks to specialized subagents:

- **MathAgent**: Mathematical operations (addition, multiplication)
- **TextAgent**: Text processing (word counting, text reversal)
- **TimeAgent**: Timezone queries (USA/China time)
- **WeatherAgent**: Weather information (city weather, temperature comparisons)

## Features

- ReAct (Reasoning + Acting) pattern for all agents
- Hierarchical agent delegation
- Real-time execution tracing
- Support for multi-part queries

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

The example uses Ollama with qwen3:1.7b. Update the configuration in `example.py`:

```python
dspy.configure(lm=dspy.LM('ollama_chat/qwen3:1.7b', api_base='http://localhost:11434', api_key=''))
```

## Usage

```python
from example import MainAgent

root_agent = MainAgent()
result = root_agent(user_query="What's the weather in Berlin and the current time in China?")
print(result.final_answer)
```

## Example Queries

- Single task: `"What's 5 + 3?"`
- Multi-task: `"Count words in 'hello world' and get USA time"`
- Complex: `"Compare temperatures between Cairo and Berlin"`

## Architecture

```
MainAgent (Coordinator)
├── MathAgent (add_numbers, multiply_numbers)
├── TextAgent (count_words, reverse_text)
├── TimeAgent (get_usa_time, get_china_time)
└── WeatherAgent (get_weather_by_city, compare_city_temperatures)
```

## Dependencies

- dspy >= 3.0.3
- pytz >= 2025.2
- requests >= 2.31.0
