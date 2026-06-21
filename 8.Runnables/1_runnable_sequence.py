from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI( model="gemini-2.5-flash")

parser = StrOutputParser()

template1 = PromptTemplate(
    template='Write a joke about {topic}.',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Explain the following joke in 2-3 lines: {joke}.',
    input_variables=['joke']
)

chain = RunnableSequence(template1, model, parser, template2, model, parser)

response  = chain.invoke({"topic":"Cricket"})

print(response)