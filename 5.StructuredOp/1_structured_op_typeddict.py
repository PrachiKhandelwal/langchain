from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Literal

load_dotenv()

class Review(TypedDict):
    summary:str
    sentiment:Literal['positive','negative','neutral']
    
model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

structured_model = model.with_structured_output(Review)
response = structured_model.invoke('The app has a simple interface and basic features are easy to use. Performance is acceptable, though some screens take a few seconds to load. It does what it claims, but there is nothing particularly unique about it.')
print(response)