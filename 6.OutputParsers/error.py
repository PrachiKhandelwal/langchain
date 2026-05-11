from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Person(BaseModel):
    name:str
    age:int=Field(ge=18)
    city:str

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(template='Generate name, age and city for a fictious person.\n {formatting_instruction}', input_variables=[],partial_variables={"formatting_instruction":parser.get_format_instructions()})

chain = template | model | parser

response = chain.invoke({})

print(response)