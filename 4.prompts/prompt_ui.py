import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

load_dotenv()


st.header('Research App')

research_paper = st.selectbox('Select Research Paper',['Attention Is All You Need','BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding','GPT-3: Language Models are Few-Shot Learners','YOLO: Unified, Real-Time Object Detection'])

summary_style = st.selectbox('Select style',['Code oriented','Maths Oriented', 'Beginner', 'Academic'])

summary_length = st.selectbox('Select length',['Short (1-2 paragraphs)','Long (5-6 paragraphs)','Medium(3-4 paragraphs)'])

template = load_prompt('template.json')


if st.button('Summarize'):
    # print(prompt)
    try:
        llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        max_new_tokens=800
        )
        
        model = ChatHuggingFace(llm=llm)
        
        chain = template | model
        
        response = chain.invoke({
        'paper_input':research_paper,
        'style_input':summary_style,
        'length_input':summary_length
        })
        
        print('model response: ',response.content)
        print('------------------------------------------------')
        st.write(response.content)
    except Exception as e:
        print(e)
        st.write(e)
    