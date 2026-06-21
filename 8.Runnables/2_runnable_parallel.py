from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI( model="gemini-2.5-flash")

parser = StrOutputParser()

template1 = PromptTemplate(
    template='Write a tweet about the topic: {topic}.',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a LinkedIn about the topic: {topic}.',
    input_variables=['topic']
)

chain = RunnableParallel({
    'tweet':RunnableSequence(template1, model, parser),
    'linkedin_post':RunnableSequence(template2, model, parser) 
})

response  = chain.invoke({"topic":"My Promotion"})

print(response)