from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
     model="gemini-3-flash-preview"
)

parser = StrOutputParser()

template = PromptTemplate(template='Generate 5 interestinng facts about {topic}',input_variables=['topic'],validate_template=True)

chain = template | model |  parser

response = chain.invoke({'topic':'stars'})

print(response)

chain.get_graph().print_ascii()