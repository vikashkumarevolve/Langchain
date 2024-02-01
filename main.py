#Integrating our code with OpenAI
import os
from constaints import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

#Intialise the streamlit frameword

st.title('Langchain with openai API')
input_text=st.text_input("Search the topic you want")

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template='tell me about {name}'
)



#Intialise openai llm
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,output_key='person',verbose=True)

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template='total income of {person}'
)
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='income')

parent_chain=SequentialChain(
    chains=[chain,chain2],input_variables=['name'],output_variables=['person','income'],verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))