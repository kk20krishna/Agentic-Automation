from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchResults


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


'''
class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
'''
