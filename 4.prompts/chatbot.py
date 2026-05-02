from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

history = [SystemMessage('You are a helpful chat assistant Joe.')]

while True:
    question = input('You:')
    if(question == 'exit'):
        break
    history.append(HumanMessage(question))
    result = model.invoke(history)
    print('AI: ',result.content)
    history.append(AIMessage(result.content))
    
print('chat history',history)