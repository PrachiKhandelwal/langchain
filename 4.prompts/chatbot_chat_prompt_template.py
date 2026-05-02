from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder

from dotenv import load_dotenv

load_dotenv()

history=[]

template = ChatPromptTemplate([('system','You are a {domain} expert'),MessagesPlaceholder('history'),('human','Explain about {topic} in 2-3 lines.')])

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break
    prompt = template.invoke({"domain":"Finance","history":history,"topic":user_input})
    print(prompt)
    result = model.invoke(prompt)
    history.append(('human',user_input))
    history.append(('ai',result.content))
    