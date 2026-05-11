from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Person(BaseModel):
    name:str=Field(description='Name of the person')
    age:int=Field(description='Age of the person',gt=0,le=18)
    city:str=Field(description='Place where the person belongs to')

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate a fictitious person with a name, age and city {input_city}.\n\n"
        "Respond with ONLY a valid JSON object. "
        "Do NOT include any explanation, code or markdown.\n\n"
        "{formatting_instruction}"
    ), 
    input_variables=["input_city"],
    partial_variables={"formatting_instruction":parser.get_format_instructions()}
    )

chain = template | model | parser

response = chain.invoke({"input_city":"BLR"})

print(response)