from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

template = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} expert."),
        ("human", "Explain about {topic}"),
    ]
)

user_input = input('You: ')

prompt = template.invoke({"domain":"Finance","topic":user_input})

result = model.invoke(prompt)
print(result.content)