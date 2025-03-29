from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, HfApiModel, LiteLLMModel

class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = "Uses semantic search to retrieve relevant party planning ideas for Alfred’s superhero-themed party at Wayne Manor."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to party planning or superhero themes.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=5  # Retrieve the top 5 documents
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Simulate a knowledge base about party planning
party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]

#print("Source Documents:")
#for doc in source_docs:
#    print(f"Content: {doc.page_content}")
#    print(f"Source: {doc.metadata['source']}")
#    print("-" * 20)


# Split the documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500 ,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)


docs_processed = text_splitter.split_documents(source_docs)
#print("Processed Documents:")
#for doc in docs_processed:
#    print(f"Content: {doc.page_content}")
#    print(f"Source: {doc.metadata['source']}")
#    print("-" * 20)


# Create a retriever using BM25 algorithm
#retriever = BM25Retriever.from_documents(
#    docs_processed,
#    search_kwargs={"k": 3},  # Return top 3 results
#)

# call retriver and test
#retriever_results = retriever.invoke("party ideas")
#print(retriever_results)
#print("Retrieval Results:")
#for doc in retriever_results:
#    print(f"Content: {doc.page_content}")
#    print(f"Source: {doc.metadata['source']}")
#    print("-" * 20)


# Create the retriever tool
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)


# Get OPENROUTER_API_KEY from .env file
import os
from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY= os.getenv("OPENROUTER_API_KEY")
#print("OPENROUTER_API_KEY", OPENROUTER_API_KEY)


# Initialize the model
model = LiteLLMModel(
    #model_id="openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free",
    model_id="openrouter/deepseek/deepseek-r1:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# Initialize the agent
agent = CodeAgent(tools=[party_planning_retriever], model=model)



# Example usage
response = agent.run(
    "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
)

print(response)



