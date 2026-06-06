from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import  PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
     model="gemini-2.5-flash"
)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Provides the sentiment of the given feedback.')
    
sentiment_parser = PydanticOutputParser(pydantic_object=Feedback)

classifier_template = PromptTemplate(
    template='Classify the sentiment of below feedback as positive or negative. \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    validate_template=True,
    partial_variables={'format_instructions':sentiment_parser.get_format_instructions()}
)

sentiment_chain = classifier_template | model | sentiment_parser

class Reply(BaseModel):
    reply: str=Field(description='Written Response to customer based on the feedback provided.')
    
reply_parser = PydanticOutputParser(pydantic_object=Reply)

postive_feedback_template = PromptTemplate(
    template='''
    You are a customer support agent.
    Customer Feedback:
    {feedback}
    Write a polite response to this positive feedback.
    {format_instructions}
    ''',
    input_variables=['feedback'],
    partial_variables={'format_instructions':reply_parser.get_format_instructions()},
    validate_template=True
)

negative_feedback_template = PromptTemplate(
    template='''
    You are a customer support agent.
    Customer Feedback:
    {feedback}
    Write a polite response to this negative feedback.
    {format_instructions}
    ''',
    input_variables=['feedback'],
    partial_variables={'format_instructions':reply_parser.get_format_instructions()},
    validate_template=True
)

branch_chain = RunnableBranch(
    (lambda x: x['sentiment'] == 'positive' , postive_feedback_template | model | reply_parser),
    (lambda x: x['sentiment'] == 'negative',negative_feedback_template | model | reply_parser),
    RunnableLambda(lambda x: 'Could not find sentiment')
)

def prep_data(x):
    sentiment = sentiment_chain.invoke(x)
    data = {
        'feedback':x['feedback'],
        'sentiment':sentiment.sentiment
    }
    return data

chain = prep_data | branch_chain

response = chain.invoke({'feedback':'Received expired product'})

print(response)

chain.get_graph().print_ascii()