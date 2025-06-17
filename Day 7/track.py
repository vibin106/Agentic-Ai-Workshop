import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set your Gemini API Key directly (for dev only)
from config import GEMINI_MODEL, EMBEDDING_MODEL

# --- PDF Loader and Chunking ---
@st.cache_resource
def load_vectorstore(pdf_path="learning_tracks_30_domains.pdf"):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Chunking the content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = text_splitter.split_documents(pages)

    # Embeddings from Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Save into Chroma vectorstore
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chromadb")
    return vectordb

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template(
    "You are a career advisor. Based on the user's query, provide concise and clear learning track recommendations.\n\nContext:\n{context}\n\nUser Query: {question}\n\nAnswer:"
)

# --- LangChain Retrieval Chain ---
def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, max_output_tokens=512)
    return (
        {"context": retriever, "question": lambda x: x["question"]}
        | prompt_template
        | llm
        | StrOutputParser()
    )

# --- Streamlit UI ---
st.set_page_config(page_title="Track Recommender Agent", layout="wide")
st.title("🎯 Track Recommender Agent (RAG-enabled)")
st.markdown("Identify learning paths for your target domain using personalized AI recommendations.")

# User Input
user_query = st.text_input("Enter your target domain or career transition query (e.g., 'I want to switch to cloud engineering'):")

if user_query:
    with st.spinner("Analyzing and retrieving recommendations..."):
        vectordb = load_vectorstore()
        chain = get_rag_chain(vectordb)
        result = chain.invoke({"question": user_query})
        st.success("✅ Recommendation:")
        st.write(result)
