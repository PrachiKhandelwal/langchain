from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
     model="gemini-3-flash-preview"
)

parser = StrOutputParser()

report_template = PromptTemplate(template='Generate a detailed report (max 15 lines) on {topic}',input_variables=['topic'],validate_template=True)

imp_pointers_template = PromptTemplate(template='Identify and list down 5 most important points from the below text.\n {text}',input_variables=['text'],validate_template=True)

chain = report_template | model | parser | imp_pointers_template | model | parser

response = chain.invoke({'topic':'French Revolution'})

print(response)