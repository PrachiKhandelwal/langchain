from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI( model="gemini-2.5-flash")

parser = StrOutputParser()

passthrough = RunnablePassthrough()

prompt1= PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke: {joke}',
    input_variables=['joke']
)

joke_generation_chain = prompt1 | model | parser

joke_explain_chain =  RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':prompt2 | model | parser
})

final_chain = joke_generation_chain | joke_explain_chain

response = final_chain.invoke({'topic':'universe'})
print(response)