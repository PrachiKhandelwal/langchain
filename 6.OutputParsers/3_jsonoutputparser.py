from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and contact number of a fictional person.\n{format_instructions}",
    # partial variables are  those which are passed before runtime only
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

json_response = chain.invoke({})

print(json_response)
print(type(json_response))