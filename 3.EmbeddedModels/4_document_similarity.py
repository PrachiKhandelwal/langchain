from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

documents =[
   'Google Cloud Next is an annual conference where Google presents its latest developments in AI, cloud infrastructure, and data analytics.',
    'The event highlights tools that help organizations build, deploy, and manage autonomous AI agents to automate complex workflows.',
    'Next 26, in Las Vegas, focuses on transitioning from AI experimentation to real-world deployment, with a focus on Agentic Data Cloud and secure-by-design AI systems'
]

query = "what is purpose of google cloud next"

documents_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

result = cosine_similarity([query_embedding],documents_embedding) # returns 2D array
indexed_result = list(enumerate(result[0]))
sorted_result = sorted(indexed_result, key=lambda x:x[1])
print(sorted_result)
