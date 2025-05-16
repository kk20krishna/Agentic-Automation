# Install required packages:
# pip install python-dotenv crewai langchain-community pydantic langtrace-python-sdk -U -q

import os
from dotenv import load_dotenv
from datetime import datetime

# Must precede any llm module imports
from langtrace_python_sdk import langtrace
langtrace.init(api_key = 'deead3b309930e2343540cfe57693978e217cc87443e5f013ce708ccea4ecc50')

from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchResults
from crewai.tools import BaseTool
from pydantic import Field
from crewai import LLM
import base64

print("$$$$$$$$$$$$$ CrewAI Research Crew $$$$$$$$$$$$$")


# Using Ollama for local inference
#os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-ollama"
#llm = "ollama/llama3.2:latest"

# using OpenRouter for inference
# set OPENROUTER_API_KEY="xxxxxxxxx" in .env file
load_dotenv()
llm = LLM(
    #model="openrouter/deepseek/deepseek-r1:free",
    model="openrouter/deepseek/deepseek-chat-v3-0324:free",
    base_url="https://openrouter.ai/api/v1",
)

class SearchTool(BaseTool):
    name: str = "News Search Tool"
    description: str = "Useful for searching the internet for the latest news. Use this to find the latest news about {topic}."
    search: DuckDuckGoSearchResults = Field(default_factory=lambda: DuckDuckGoSearchResults(backend="news"))

    def _run(self, query: str) -> str:
        """Execute the search query and return the latest news."""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"


search_agent = Agent(
    role="Senior Search Associate",
    goal="Search the internet for the latest news about {topic} and provide it to the Senior Content Writer.",
    backstory=(
        "You love staying informed and are an expert in finding the latest news online. "
        "Your job is to find as much up-to-date information as possible and provide it to the Senior Content Writer."
    ),
    llm=llm,
    tools=[SearchTool()],
    verbose=True
)

content_writer_agent = Agent(
    role="Senior Content Writer",
    goal="Write an article using the latest news on {topic} provided by the Senior Search Associate.",
    backstory=(
        "You are a highly experienced content writer who has worked for major media organizations. "
        "You specialize in writing accurate and well-cited news articles, ensuring that all information is up-to-date."
        "You make sure that only the latest new is used in your articles."
        "You are also responsible for citing all sources properly."
        "You will only use the information provided by the Senior Search Associate."
        "You will always cite the source of the news."
    ),
    llm=llm,
    verbose=True
)

editor_agent = Agent(
    role="Senior Editor",
    goal="Create an editorial opinion based on the news article created by the Senior Content Writer on {topic}.",
    backstory=(
        "You are a respected senior editor at a leading Indian newspaper. "
        "Your editorials provide insightful analysis of how global events impact India. "
        "You ensure that opinions are balanced, accurate, and forward-looking."
        "You will only use the information provided by the Senior Content Writer."
    ),
    llm=llm,
    verbose=True
)

search_task = Task(
    description="Find the latest news about {topic}.",
    expected_output="Latest news about {topic} with news source citations. Provide as much news as possible.",
    agent=search_agent,
    verbose=True
)

write_task = Task(
    description="Write a four-paragraph news article on {topic} based on the latest news given by the Senior Search Associate. Cite news sources.",
    expected_output="A four-paragraph news article on {topic} with news source citations. Each paragraph should have 3-4 sentences.",
    agent=content_writer_agent,
    verbose=True
)

edit_task = Task(
    description=(
        "Write a two-paragraph India-focused editorial opinion on {topic} and append it to the news article. "
        "Each paragraph should have 3-4 sentences. "
        "The editorial opinion should be based on the latest news, be concise, and offer insightful analysis. "
        "It should highlight the relevance of the news to India and how India should react to this news."
    ),
    expected_output="A four-paragraph news article (with citations) on {topic} followed by a two-paragraph India-focused editorial opinion.",
    agent=editor_agent,
    verbose=True
)

crew = Crew(
    agents=[search_agent, content_writer_agent, editor_agent],
    tasks=[search_task, write_task, edit_task],
    verbose=True
)

inputs = {"topic": "Trump and Zelensky"}

print("\n############ Kicking off Editorial Opinion Crew ##############")
print("Topic:", inputs["topic"])
print("##############################################################\n")

result = crew.kickoff(inputs=inputs)

print("\n############ Final News Article with India-Focused Editorial Opinion ##############")
print(result)

# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Construct the filename with date & time
file_path = f'CrewAI NewsRoom Crew\\Generated_Editorial_Opinions\\editorial_opinion_{inputs["topic"]}_{current_time}.txt'

# Write the result to the file
with open(file_path, "w", encoding="utf-8") as file:
    file.write(str(result))