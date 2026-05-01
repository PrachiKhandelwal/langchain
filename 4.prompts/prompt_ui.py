import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate(template='''
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:

   * Include relevant mathematical equations if present in the paper.
   * Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:

   * Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provided style and length. End with a complete concluding sentence. Do not stop mid-sentence. If you are running out of space, wrap up early rather than cutting off.

''', 
input_variables=['paper_input','style_input','length_input'],
validate_template=True
)

st.header('Research App')

research_paper = st.selectbox('Select Research Paper',['Attention Is All You Need','BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding','GPT-3: Language Models are Few-Shot Learners','YOLO: Unified, Real-Time Object Detection'])

summary_style = st.selectbox('Select style',['Code oriented','Maths Oriented', 'Beginner', 'Academic'])

summary_length = st.selectbox('Select length',['Short (1-2 paragraphs)','Long (5-6 paragraphs)','Medium(3-4 paragraphs)'])

prompt = template.invoke({
    'paper_input':research_paper,
    'style_input':summary_style,
    'length_input':summary_length
})

if st.button('Summarize'):
    # print(prompt)
    try:
        llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        max_new_tokens=800
        )

        model = ChatHuggingFace(llm=llm)
        response = model.invoke(prompt)
        print('model response: ',response.content)
        print('------------------------------------------------')
        st.write(response.content)
    except Exception as e:
        print(e)
        st.write(e)
    