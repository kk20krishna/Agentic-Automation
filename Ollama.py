from ollama import Client

client = Client(host='http://localhost:11434')

model="llama3.2:latest"

prompt = "What is the capital of France?"

response = client.generate(model=model, prompt=prompt)

print(response['response'])