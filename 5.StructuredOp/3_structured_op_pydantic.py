from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()

class Review(BaseModel):
    key_themes:list[str]=Field(description='Write down all key themes discussed in the review')
    summary:str=Field(description='1-2 liner summary of the review')
    sentiment:Literal['positive','negative','neutral']=Field(description='Sentiment of the review') 
    product:str= Field(description='Name of the reviewed product')
    author:Optional[str] = Field(default=None,description='Name of the review author')
    cons:Optional[list[str]]=Field(default=None,description='Negative points mentioned in review in form of list')
    pros:Optional[list[str]]  = Field(default=None,description='Positive points mentioned in review in form of list')

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

structured_model = model.with_structured_output(Review)

response = structured_model.invoke("I've been using the Samsung Galaxy Watch 7 for the past few weeks, and overall I'm very impressed with it. The display is bright and smooth, notifications arrive instantly, and the health tracking features like heart rate and sleep monitoring are genuinely useful in daily life. Battery life easily lasts more than a day, although heavy GPS usage drains it faster than expected. The watch feels premium and comfortable to wear, but the charging speed could still be better for a device in this price range. Performance is fast, apps open quickly, and fitness tracking during workouts has been accurate so far. While some extra Samsung apps feel unnecessary, the overall experience is polished, reliable, and definitely worth recommending for Android users.")

print(response)