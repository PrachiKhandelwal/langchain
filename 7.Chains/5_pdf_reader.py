from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import faiss

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

# creates loader for text file
loader = TextLoader('sample.txt')

# reads file contents
documents = loader.load()

# create text splitter object
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)

# split documents into chunks
texts = text_splitter.split_documents(documents)

# create embedding model to convert text to vector
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# to test latest embedding model bug
"""
result = embeddings.embed_documents(["hello", "world", "test"])
print(f"Number of embeddings returned: {len(result)}")
"""

# create empty vector container
index = faiss.IndexFlatL2(len(embeddings.embed_query('hello world')))

vector_store = FAISS(
    # indicates the model to be used to convert text into embedding
    embedding_function=embeddings,
    
    # store and search vector using this FAISS index
    index=index,
    # stores original text
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(texts)

query = input('Enter your question: ')

retrieved_docs = vector_store.similarity_search(query=query, k=2)

context = '\n'.join(doc.page_content for doc in retrieved_docs)

template = PromptTemplate(template='Answer the question using given context only. If the answer is not  present in the context, say you do not know the answer. \n Question: {question} \n Context: {context}', input_variables=['question','context'],validate_template=True)

chain = template | model | parser

response = chain.invoke({'question': query, 'context':context})

print(response)