from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# MessagesPlaceholder is used to rerieve and store chat history for model's use

load_dotenv()

history  = []

with open('messages.txt') as f:
    history.extend(f.readlines())

template = ChatPromptTemplate([('system','You are a helpful support agent.'), MessagesPlaceholder("history"),('human','{user_query}')])

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

user_input = input('How can I help you?')
prompt = template.invoke({
    "history":history,
    "user_query":user_input
})
print(prompt)
response = model.invoke(prompt)
print(response.content)