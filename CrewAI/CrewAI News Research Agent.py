import os
from crewai import Agent, Task, Crew

os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-ollama"

llm='ollama/llama3.2:latest'

info_agent = Agent(
    role="Information Agent",
    goal="Give compelling information about a certain topic",
    backstory="""
        You love to know information.  People love and hate you for it.  You win most of the
        quizzes at your local pub.
    """,
    llm=llm
)

review_agent = Agent(
    role="Review Agent",
    goal="Review the information given by the information agent",
    backstory="""
        You are a very critical person.  You are very good at finding flaws in information.
        You are very good at making sure that information is correct.
        You will review the information and provide feedback.
        You will make sure that the information is accurate and correct.
        You will ensure that the information is given in the form of a rhyme.
    """,
    llm=llm
)


task1 = Task(
    description="Tell me all about the box order management system.",
    expected_output="Give me a quick summary and then also give me 7 bullet points describing it.",
    agent=info_agent
)

task2 = Task(
    description="Review the information given by the Information Agent.",
    expected_output="Give feedback on the information given by the information agent.",
    agent=review_agent
)


crew = Crew(
    agents=[info_agent, review_agent],
    tasks=[task1, task2],
    verbose=True
)

result = crew.kickoff()

print("############")
print(result)