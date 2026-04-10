import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_agent
from langchain_core.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq

import torch

# ---------------- UI ----------------
st.set_page_config(page_title="AI Vakeel", layout="wide")

img_path = "images.jpg"

col1, col2, col3 = st.columns([1,2,1])

with col2:
    if os.path.exists(img_path):
        st.image(img_path, width=900)
    else:
        st.warning("Image not found")

st.title("AI Vakeel - Indian Law Assistant")

# ---------------- MODEL ----------------
groq = ChatGroq(model='llama-3.1-8b-instant')

# ---------------- LOAD EXISTING VECTOR DB ONLY ----------------
@st.cache_resource
def load_vectorstore():

    hf_embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    vector_store = FAISS.load_local(
        "faiss_law_db",
        hf_embedding,
        allow_dangerous_deserialization=True
    )

    return vector_store

vector_store = load_vectorstore()

retriver = vector_store.as_retriever(search_kwargs={"k": 10})

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template("""
You are an expert AI lawyer specializing in Indian law.

Answer only from context.

Context:
{context}

Question:
{question}
""")

# ---------------- TOOLS ----------------
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
duckduckgo = DuckDuckGoSearchResults()

retriever_tool = create_retriever_tool(
    retriver,
    "indian_law_rag",
    "Search Indian law documents"
)

tools = [retriever_tool, wikipedia, duckduckgo]

agent = create_agent(
    model=groq,
    tools=tools
)

# ---------------- INPUT ----------------
question = st.text_input("Enter your legal question")

if st.button("Ask"):

    if question.strip() != "":

        retrieved_docs = retriver.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": question
        })

        rag_response = groq.invoke(final_prompt)

        if "I could not find" not in rag_response.content:
            source = "RAG"
            answer = rag_response.content
        else:
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            source = "AGENT"
            answer = result["messages"][-1].content

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Source")
        st.write(source)

# ---------------- FOOTER ----------------
st.markdown("""
<br>
<div style='text-align:center; padding:12px; background-color:#111111; border-radius:10px;'>
<span style='color:#AAAAAA; font-size:16px;'>
Designed & Developed by <b style='color:#CCCCCC;'>Yedeedya Injeti</b><br>
Under <b style='color:#B8860B;'>Innomatics Research Labs</b>
</span>
</div>
<br>
""", unsafe_allow_html=True)