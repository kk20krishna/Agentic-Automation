from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

# Get OPENROUTER_API_KEY from .env file
import os
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY= os.getenv("OPENROUTER_API_KEY")
print("OPENROUTER_API_KEY", OPENROUTER_API_KEY)


# Initialize the search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the model
model = LiteLLMModel(
    model_id="openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

agent = CodeAgent(
    model = model,
    tools=[search_tool]
)

# Example usage
response = agent.run(
    "Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering."
)
print(response)