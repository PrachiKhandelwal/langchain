from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents
docs = [
    "React.js is a JavaScript framework.",
    "LangChain is a framework for building context-aware reasoning applications.",
    "LangGraph is related to LangChain."
]

# Vector store
vectorstore = InMemoryVectorStore.from_texts(
    docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Dynamic question
question = input("Ask your question: ")

# Retrieve relevant docs
retrieved_docs = retriever.invoke(question)

# Combine retrieved text
context = "\n".join([doc.page_content for doc in retrieved_docs])

# Prompt LLM
prompt = f"""
Answer the question only using the context below.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)