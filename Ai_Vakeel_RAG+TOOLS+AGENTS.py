

# # **Ai Vakeel 🎗⚖️** 


# ### **1. RAG**


# ### 1. Importing requried lib


import langchain
from IPython.display import Markdown
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import HumanMessagePromptTemplate,SystemMessagePromptTemplate,AIMessagePromptTemplate,PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_agent
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.tools.retriever import create_retriever_tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ### 2. Load Model


groq = ChatGroq(model = 'llama-3.1-8b-instant')

groq


gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)

# ### 3. Loading the documents & RAG implementation


import os
print(os.getenv("GOOGLE_API_KEY"))

# ##### 3.1 - Load documents


# - https://judgments.ecourts.gov.in/pdfsearch/?p=pdf_search/home&text=High%20Court&captcha=aLb2yu&search_opt=PHRASE&fcourt_type=2&escr_flag=&proximity=&sel_lang=&app_token=b1ce681a918db1a9adf28ca0842f8c5c32869b3fb4bdff60bad815e201b7aa0b


# files = [
#     "data/Constitution of India.pdf",
#     "data/Indian Contract Act, 1872.pdf",
#     "data/Indian Evidence Act.pdf",
#     "data/Indian Penal Code (IPC).pdf"
# ]


import os

files = []

for file in os.listdir("data"):
    if file.lower().endswith(".pdf"):
        files.append(os.path.join("data", file))

docs = []

for file in files:
    loader = PyPDFLoader(file)
    loaded_docs = loader.load()
    docs.extend(loaded_docs)

len(docs)

# **loaded_docs = loader.load()**
# - Reads the PDF and loads its content.
# - Usually, each page of the PDF is stored as a separate Document object.


type(docs)

# ##### 3.2 - Text Spliting


# creating the text spliter
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 300,
    separators=["\n\n", "\n", ".", " ", ""]
)
# creating the chunks using the split_document
pdf_chuncks = text_spliter.split_documents(docs)

print(pdf_chuncks[0].page_content)

print(pdf_chuncks[0].page_content)

print("total chunks :", len(pdf_chuncks))
print()
print("first 5 chunks")
print()

for i in range(5):
    print(f"chunk {i+1}")
    print(pdf_chuncks[i].page_content)
    print("-" * 100)

for i in range(len(pdf_chuncks)):
    print(f"chunk {i+1}")
    print(pdf_chuncks[i].page_content)
    print("-" * 100)
    

len(pdf_chuncks)

# - Total 54747 chuncks are created


len(docs)

# - with 4 pdf files 634 documents are created


# ##### 3.3 - Create MetaData


# create metadata
for i in range(len(pdf_chuncks)):
    pdf_chuncks[i].metadata["chunk_id"] = i + 1
    pdf_chuncks[i].metadata["chunk_size"] = len(pdf_chuncks[i].page_content)
    pdf_chuncks[i].metadata["document_type"] = "law_pdf"
    pdf_chuncks[i].metadata["file_name"] = pdf_chuncks[i].metadata["source"].split("/")[-1]

# print metadata
print(pdf_chuncks[0].metadata)

# print first 5 metadata
for i in range(5):
    print(f"chunk {i+1} metadata")
    print(pdf_chuncks[i].metadata)
    print("-" * 100)

# ##### 3.4 Creating embedings


# **First Important Point**
# 
# - Groq does NOT provide embeddings models (as of now)
# 
# 
# - Groq → for LLM (generation)
# - Other provider → for embeddings
# 
# - so the work flow is:
# 
# - **PDFs -> chunks -> Hugging Face embeddings -> FAISS → retriever -> Groq LLM**


from langchain_huggingface import HuggingFaceEmbeddings
import torch

hf_embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"batch_size": 16}
)

print("device:", "cuda" if torch.cuda.is_available() else "cpu")
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# ##### 3.5 Creating the database Faiss


vector_db_path = "faiss_law_db"

if os.path.exists(vector_db_path):
    vector_store = FAISS.load_local(
        vector_db_path,
        hf_embedding,
        allow_dangerous_deserialization = True
    )
    print("vector db loaded successfully")

else:
    vector_store = FAISS.from_documents(pdf_chuncks, hf_embedding)
    vector_store.save_local(vector_db_path)
    print("vector db created and saved successfully")

# ##### 3.6 - Creating the retriver


retriver = vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':10}
)

question = 'attempt murder punishment'

question = "Section 437 IPC"

search_query = question + " exact section Indian Penal Code bail mischief with fire"

retrieved_docs = retriver.invoke(question)
retrieved_docs

# ### 4. Prompt_template


prompt = ChatPromptTemplate.from_template("""
You are an expert AI lawyer specializing in Indian law.

You must answer ONLY from the provided legal context.

Strict legal answering rules:
1. If the question mentions a specific section number, use only that exact section number from the context.
2. If that exact section number is not present in the context, do not answer from any other section.
3. If the question mentions an offence name instead of a section number, identify the exact section that directly defines that offence.
4. Always prefer a specific offence section over a general section.
5. Never substitute a nearby or similar section for the exact one asked.
6. Never combine punishments from multiple sections.
7. Never use outside knowledge.
8. If the exact section is missing, say exactly:
   "I could not find the exact legal section in the retrieved context."

Context:
{context}

Question:
{question}

Return the answer in exactly this format:

Relevant Act:
Section:
Offence:
Punishment:
Explanation:
""")

# CHAT LOOP
'''
print(" AI Lawyer — Indian Law Assistant")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # retrieve documents
    retrieved_docs = retriver.invoke(question)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = gemini.invoke(final_prompt)

    print(" AI Lawyer:\n")
    print(response.content)
    print()'''



# ### 5. Tools


# ### 4. Tools

wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia)

duckduckgo_tool = DuckDuckGoSearchResults(
    num_results=5
)

retriever_tool = create_retriever_tool(
    retriver,
    "indian_law_rag",
    "Searches the loaded Indian law PDFs and returns relevant legal sections, punishments, and explanations."
)

tools = [retriever_tool, wikipedia_tool, duckduckgo_tool]

# ### 6. Creating agent


# ### 5. Create Agent

agent = create_agent(
    model=gemini,
    tools=tools,
    system_prompt="""
You are an expert AI lawyer specializing in Indian law.

You have 3 sources:
1. indian_law_rag -> use first for Indian law sections, punishments, bare acts, constitutional provisions, contract law, evidence law, criminal law.
2. wikipedia -> use only if indian_law_rag does not contain enough information and the user asks for general legal background or concept explanation.
3. duckduckgo_search_results_json -> use for latest, current, recent, fresh judgments, legal news, or recent Indian court developments.

Strict tool rules:
1. Always try indian_law_rag first.
2. If the answer is found clearly in indian_law_rag, answer from it and do not use wikipedia or duckduckgo.
3. If indian_law_rag does not contain enough relevant information, then use wikipedia for general explanation.
4. If the question asks about latest, current, recent, new, today, fresh judgments, recent Supreme Court, recent High Court, or current legal updates, use duckduckgo_search_results_json.
5. Prefer Indian legal and court-related results when using web search.
6. For exact section-number questions, prefer indian_law_rag and do not replace the asked section with nearby sections.
7. Do not mix answers from unrelated sections.
8. If exact legal text is not found, say so clearly.
9. Keep the answer clear and structured.

If you use duckduckgo for legal judgments, prefer results related to:
- sci.gov.in
- judgments.ecourts.gov.in
- hcservices.ecourts.gov.in
- indiankanoon.org
- indiacode.nic.in
"""
)

# ### 7. Texting 


print("Ai Vakeel 🎗⚖️ — Indian Law Assistant")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    if "section" in question.lower() and "ipc" in question.lower():
        sec_no = "".join([ch for ch in question if ch.isdigit()])
        search_query = f"{sec_no} Indian Penal Code section {sec_no} IPC exact legal text"
        retrieved_docs = retriver.invoke(search_query)
    else:
        retrieved_docs = retriver.invoke(question)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    rag_response = gemini.invoke(final_prompt)

    if "I could not find the exact legal section in the retrieved context." not in rag_response.content:
        source_name = "RAG"
        final_answer = rag_response.content
    else:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        source_name = "TOOLS / AGENT"
        final_answer = result["messages"][-1].content

    display(Markdown(f"""
# AI Lawyer Response

## Question
{question}

## Information Source
{source_name}

## Answer
{final_answer}
"""))

# ### 8. Complete image


import streamlit as st
import langchain
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import HumanMessagePromptTemplate,SystemMessagePromptTemplate,AIMessagePromptTemplate,PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_agent
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.tools.retriever import create_retriever_tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import torch
from IPython.display import Markdown

st.set_page_config(page_title="Ai Vakeel", page_icon="⚖️", layout="wide")
st.title("Ai Vakeel 🎗⚖️ — Indian Law Assistant")

# ### 2. Load Model

groq = ChatGroq(model='llama-3.1-8b-instant')

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)

# ### 3. Loading the documents & RAG implementation

files = []

for file in os.listdir("data"):
    if file.lower().endswith(".pdf"):
        files.append(os.path.join("data", file))

docs = []

for file in files:
    loader = PyPDFLoader(file)
    loaded_docs = loader.load()
    docs.extend(loaded_docs)

# ##### 3.2 - Text Spliting

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ".", " ", ""]
)

pdf_chuncks = text_spliter.split_documents(docs)

# ##### 3.3 - Create MetaData

for i in range(len(pdf_chuncks)):
    pdf_chuncks[i].metadata["chunk_id"] = i + 1
    pdf_chuncks[i].metadata["chunk_size"] = len(pdf_chuncks[i].page_content)
    pdf_chuncks[i].metadata["document_type"] = "law_pdf"
    pdf_chuncks[i].metadata["file_name"] = pdf_chuncks[i].metadata["source"].split("/")[-1]

# ##### 3.4 Creating embedings

hf_embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"batch_size": 16}
)

# ##### 3.5 Creating the database Faiss

vector_db_path = "faiss_law_db"

if os.path.exists(vector_db_path):
    vector_store = FAISS.load_local(
        vector_db_path,
        hf_embedding,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = FAISS.from_documents(pdf_chuncks, hf_embedding)
    vector_store.save_local(vector_db_path)

# ##### 3.6 - Creating the retriver

retriver = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 10}
)

# ### 4. Prompt_template

prompt = ChatPromptTemplate.from_template("""
You are an expert AI lawyer specializing in Indian law.

You must answer ONLY from the provided legal context.

Strict legal answering rules:
1. If the question mentions a specific section number, use only that exact section number from the context.
2. If that exact section number is not present in the context, do not answer from any other section.
3. If the question mentions an offence name instead of a section number, identify the exact section that directly defines that offence.
4. Always prefer a specific offence section over a general section.
5. Never substitute a nearby or similar section for the exact one asked.
6. Never combine punishments from multiple sections.
7. Never use outside knowledge.
8. If the exact section is missing, say exactly:
   "I could not find the exact legal section in the retrieved context."

Context:
{context}

Question:
{question}

Return the answer in exactly this format:

Relevant Act:
Section:
Offence:
Punishment:
Explanation:
""")

# ### 5. Tools

wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia)

duckduckgo_tool = DuckDuckGoSearchResults(
    num_results=5
)

retriever_tool = create_retriever_tool(
    retriver,
    "indian_law_rag",
    "Searches the loaded Indian law PDFs and returns relevant legal sections, punishments, and explanations."
)

tools = [retriever_tool, wikipedia_tool, duckduckgo_tool]

# ### 6. Creating agent

agent = create_agent(
    model=gemini,
    tools=tools,
    system_prompt="""
You are an expert AI lawyer specializing in Indian law.

You have 3 sources:
1. indian_law_rag -> use first for Indian law sections, punishments, bare acts, constitutional provisions, contract law, evidence law, criminal law.
2. wikipedia -> use only if indian_law_rag does not contain enough information and the user asks for general legal background or concept explanation.
3. duckduckgo_search_results_json -> use for latest, current, recent, fresh judgments, legal news, or recent Indian court developments.

Strict tool rules:
1. Always try indian_law_rag first.
2. If the answer is found clearly in indian_law_rag, answer from it and do not use wikipedia or duckduckgo.
3. If indian_law_rag does not contain enough relevant information, then use wikipedia for general explanation.
4. If the question asks about latest, current, recent, new, today, fresh judgments, recent Supreme Court, recent High Court, or current legal updates, use duckduckgo_search_results_json.
5. Prefer Indian legal and court-related results when using web search.
6. For exact section-number questions, prefer indian_law_rag and do not replace the asked section with nearby sections.
7. Do not mix answers from unrelated sections.
8. If exact legal text is not found, say so clearly.
9. Keep the answer clear and structured.

If you use duckduckgo for legal judgments, prefer results related to:
- sci.gov.in
- judgments.ecourts.gov.in
- hcservices.ecourts.gov.in
- indiankanoon.org
- indiacode.nic.in
"""
)

# ### 7. Streamlit UI

question = st.text_input("Enter your legal question")

if st.button("Ask"):

    if question.strip() != "":

        if "section" in question.lower() and "ipc" in question.lower():
            sec_no = "".join([ch for ch in question if ch.isdigit()])
            search_query = f"{sec_no} Indian Penal Code section {sec_no} IPC exact legal text"
            retrieved_docs = retriver.invoke(search_query)
        else:
            retrieved_docs = retriver.invoke(question)

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": question
        })

        rag_response = gemini.invoke(final_prompt)

        if "I could not find the exact legal section in the retrieved context." not in rag_response.content:
            source_name = "RAG"
            final_answer = rag_response.content
        else:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            source_name = "TOOLS / AGENT"
            final_answer = result["messages"][-1].content

        st.markdown(f"""
# AI Lawyer Response

## Question
{question}

## Information Source
{source_name}

## Answer
{final_answer}
""")