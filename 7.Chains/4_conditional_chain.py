from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
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
    
class Reply(BaseModel):
    reply: str=Field(description='Written Response to customer based on the feedback provided.')
    
sentiment_parser = PydanticOutputParser(pydantic_object=Feedback)
reply_parser = PydanticOutputParser(pydantic_object=Reply)

classifier_template = PromptTemplate(
    template='Classify the sentiment of below feedback as positive or negative. \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    validate_template=True,
    partial_variables={'format_instructions':sentiment_parser.get_format_instructions()}
)

sentiment_chain = classifier_template | model | sentiment_parser

postive_feedback_template = PromptTemplate(
    template='Write an appropriate feedback for positive feedback as a customer agent. \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':reply_parser.get_format_instructions()},
    validate_template=True
)

negative_feedback_template = PromptTemplate(
    template='Write an appropriate feedback for negative feedback as a customer agent. \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':reply_parser.get_format_instructions()},
    validate_template=True
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive' , postive_feedback_template | model | reply_parser),
    (lambda x: x.sentiment == 'negative', negative_feedback_template | model | reply_parser),
    RunnableLambda(lambda x: 'Could not find sentiment')
)

chain = sentiment_chain | branch_chain

response = chain.invoke({'feedback':'Ugly'})

print(response)

chain.get_graph().print_ascii()