# converts any python function into a runnable
# can be used for pre processing, transformation, API calls, filtering, post processing


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

def word_count(text):
    return len(text.split(' '))

model = ChatGoogleGenerativeAI( model="gemini-2.5-flash")

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Gnerate a joke on the {topic}',
    input_variables=['topic']
)

joke_gen_chain = prompt | model | parser

final_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    # 'count':RunnableLambda(lambda joke: len(joke.split(' ')))
    'count':RunnableLambda(word_count)
})

chain = joke_gen_chain | final_chain

response = chain.invoke({'topic':'AI'})

print(response)