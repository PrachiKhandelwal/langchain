from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()


json_schema={
    "title":"Review",
    "type":"object",
    "properties":{
        "key_themes":{
            "type":"array",
            "items":{
                "type":"string"
            },
            "description":'Write down all key themes discussed in the review'
        },
        "summary":{
            "type":"string",
            "description":'1-2 liner summary of the review'
        },
        "sentiment":{
            "type":"string",
            "enum":['positive','negative','neutral'],
            "description":'Sentiment of the review'
        },
        "product":{
            "type":"string",
            "description":'Name of the reviewed product'
        },
        "author":{
            "type":["string","null"],
            "description":'Name of the review author'
        },
        "cons":{
            "type":["array","null"],
            "items":{
                "type":"string"
            },
            "description":'Negative points mentioned in review in form of list'
        },
        "pros":{
            "type":["array","null"],
            "items":{
                "type":"string"
            },
            "description":'Positive points mentioned in review in form of list'
        }
    }
}

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

structured_model = model.with_structured_output(json_schema)

response = structured_model.invoke("I've been using the Samsung Galaxy Watch 7 for the past few weeks, and overall I'm very impressed with it. The display is bright and smooth, notifications arrive instantly, and the health tracking features like heart rate and sleep monitoring are genuinely useful in daily life. Battery life easily lasts more than a day, although heavy GPS usage drains it faster than expected. The watch feels premium and comfortable to wear, but the charging speed could still be better for a device in this price range. Performance is fast, apps open quickly, and fitness tracking during workouts has been accurate so far. While some extra Samsung apps feel unnecessary, the overall experience is polished, reliable, and definitely worth recommending for Android users.")

print(response)