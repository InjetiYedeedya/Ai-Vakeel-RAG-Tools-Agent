<img width="1817" height="688" alt="rag" src="https://github.com/user-attachments/assets/59876d63-4514-4b9e-aeaa-062b7a8fabca" />
<img width="1919" height="916" alt="image" src="https://github.com/user-attachments/assets/fed25d06-2a5d-47e7-851f-8e90a5268a1c" />
Ai Vakeel - Indian Law Assistant

Ai Vakeel is an AI-powered legal assistant built for Indian law. It helps you find relevant legal sections, punishments, and explanations from major Indian legal acts.

---

What it does

- Answers legal questions in plain language
- Searches through Indian law PDFs using RAG
- Falls back to web tools if the answer is not found in the database
- Finds recent court judgments using DuckDuckGo
- Returns structured answers with act name, section, offence, punishment, and explanation

---

How it works

- Loads PDF files of major Indian legal acts
- Splits documents into chunks of 1500 characters with 300 character overlap
- Converts chunks into vector embeddings using HuggingFace BAAI bge-large-en-v1.5
- Stores embeddings in a FAISS vector database locally
- Retrieves top 10 most relevant chunks for each question
- Sends retrieved context to Gemini 2.5 Flash Lite for a structured answer
- If RAG fails, an agent with three tools takes over automatically

---

Tech Stack

- LLM for answering: Gemini 2.5 Flash Lite
- Backup LLM: Llama 3.1 8B Instant via Groq
- Embeddings: HuggingFace BAAI bge-large-en-v1.5
- Vector Store: FAISS (saved locally)
- Framework: LangChain and LangChain Classic
- UI: Streamlit

---

Project Structure

- data folder contains your Indian law PDF files
- faiss-law-db folder is auto-generated and stores the vector database
- .env file stores your API keys
- main notebook is Ai-Vakeel-RAG-TOOLS-AGENTS.ipynb
- Streamlit app entry point is app.py

---

Getting Started

- Clone the repository
- Install all required packages from requirements.txt
- Create a .env file and add your GOOGLE-API-KEY and GROQ-API-KEY
- Place Indian law PDF files inside the data folder
- Run the app using: streamlit run app.py
- The first run builds the FAISS database which may take a few minutes

---

Answer Format

Every response follows this fixed structure:

- Relevant Act
- Section
- Offence
- Punishment
- Explanation

---

Agent Tool Priority

- RAG database is always checked first for bare act text
- Wikipedia is used only if RAG does not have enough information
- DuckDuckGo is used for recent judgments and current legal news
- Preferred sources for web search are indiankanoon.org, sci.gov.in, and ecourts.gov.in

---

Notes

- Vector database is saved locally so subsequent runs are much faster
- GPU acceleration is used automatically if CUDA is available
- The system never mixes sections or substitutes nearby sections for the one asked
- Exact section number questions are handled with strict matching logic

---






