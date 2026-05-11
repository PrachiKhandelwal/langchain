from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

report_template = PromptTemplate(
    template='Write a detailed report on {topic} upto 15 lines',
    input_variables=["topic"],
    validate_template=True
)

summary_template = PromptTemplate(
    template='Write a 5 line summary for following text.\n {text}',
    input_variables=["text"],
    validate_template=True
)

report_prompt = report_template.invoke({
    "topic":"black hole"
})

report_response = model.invoke(report_prompt)

summary_prompt = summary_template.invoke({
    "text": report_response.content
})

summary_response = model.invoke(summary_prompt)

print(parser.invoke(summary_response))