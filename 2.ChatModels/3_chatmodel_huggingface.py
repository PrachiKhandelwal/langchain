from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    # max_new_tokens=512,
    # do_sample=False,
    # repetition_penalty=1.03,
    # provider="auto",
)

model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke('what is capital of usa?')
    print(result.content)
except Exception as e:
    print(e)
