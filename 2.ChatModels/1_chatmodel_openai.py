from openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

result = model.invoke("What is capital of India",temperature=0, max_completion_tokens=100)

print(result.content)