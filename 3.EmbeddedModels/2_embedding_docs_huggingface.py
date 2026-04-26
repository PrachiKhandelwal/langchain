from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from dotenv import load_dotenv

load_dotenv()

'''
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

docs =  [
        "Today is Monday",
        "Today is Tuesday",
        "Today is April Fools day",
    ]
query_result = embeddings.embed_documents(docs)
print(query_result)
'''

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

text = ["React.js is  a JS framework", "LangChain is the framework for building context-aware reasoning applications", "Langgraph is related to langchain"]

vectorstore = InMemoryVectorStore.from_texts(
    text,
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Retrieve the most similar text
retrieved_documents = retriever.invoke("Define LangChain")

# show the retrieved document's content
print(retrieved_documents)