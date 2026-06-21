from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

class Category(BaseModel):
    type:Literal['Refund Request', 'Complaint', 'General Query']= Field(description='Category of given user query')

model = ChatGoogleGenerativeAI( model="gemini-2.5-flash")
structured_model = model.with_structured_output(Category)

prompt = PromptTemplate(
    template="""
    You are a customer support classifier.

    Classify the user query into exactly one category:

    1. Refund Request - requests a refund, asks refund status, refund progress, refund eligibility.
    2. Complaint - reports dissatisfaction, problems, issues, bad service.
    3. General Query - all other questions.

    User Query:
    {query}
    """,
    input_variables=['query']
)

chain = prompt | structured_model

routing_chain = RunnableBranch(
    (lambda category: category.type == 'Refund Request', RunnableLambda(lambda x: 'Route to refund agent')),
    (lambda category: category.type ==  'Complaint',RunnableLambda(lambda x:'Route to customer support team')),
    RunnableLambda(lambda x:'Route to chat agent')
)

final_chain = chain | RunnableParallel({
    'query': RunnableLambda(lambda x: x.type),
    'response':routing_chain
})

response = final_chain.invoke({'query':'what is my refund status?'})

print(response)