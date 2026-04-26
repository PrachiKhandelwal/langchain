# https://docs.langchain.com/oss/python/integrations/providers/google

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",max_output_tokens=100)

try:
    response = model.invoke("what is capital of usa")
    print(response.content)
except Exception as e:
    print(e)
    