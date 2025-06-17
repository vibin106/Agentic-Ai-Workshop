import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from config import GEMINI_MODEL, EMBEDDING_MODEL

st.set_page_config(page_title="Cross-Domain AI Navigator", layout="centered")
st.title("🧭 Cross-Domain AI Navigator")
st.markdown("Helping you pivot careers through intelligent AI assistance.")

# --- Session State to share input ---
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "structured_profile" not in st.session_state:
    st.session_state.structured_profile = ""



# --- Helper Functions ---

def get_llm():
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)

def parse_background(text):
    prompt = PromptTemplate.from_template("""
You are an AI assistant that extracts a structured professional profile from user text.

Text: {user_background}

Return as JSON:
- Education
- Previous Roles
- Years of Experience
- Career Interest
""")
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"user_background": text})

def map_skills(profile_json):
    prompt = PromptTemplate.from_template("""
You are a Skill Mapping Agent. Based on this professional profile:
{profile}

Map current skills to a new career interest and output 3 sections:
- Transferable Skills
- Obsolete Skills
- Skills Needing Upgrade
""")
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"profile": profile_json})

def unlearning_advice(profile_json, target_domain):
    prompt = PromptTemplate.from_template("""
You are an AI Unlearning Advisor. Given the user's background and career goal, list 5–7 things they should unlearn.

Profile: {profile}
Target Domain: {target}
""")
    chain = prompt | ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.5, max_output_tokens=700) | StrOutputParser()
    return chain.invoke({"profile": profile_json, "target": target_domain or "new domain"})

def recommend_tracks(query):
    pdf_path = "learning_tracks.pdf"
    loader = PyPDFLoader("learning_tracks_30_domains.pdf")
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectordb.as_retriever()

    relevant_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs[:3])

    prompt = PromptTemplate.from_template("""
You are a Track Recommender AI. Use the following context to suggest learning paths, certifications, or skill-building steps:

Context:
{context}

User Goal:
{query}

Respond with 3-5 personalized learning recommendations.
""")

    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"context": context, "query": query})

# --- Tabs ---
tabs = st.tabs(["🧾 Background Parser", "🔍 Skill Mapping", "🧠 Unlearning Advisor", "📚 Track Recommender"])

# --- TAB 1: Background Parser ---
with tabs[0]:
    st.subheader("Background Parsing Agent")
    input_text = st.text_area("Describe your career background (education, roles, interest):", value=st.session_state.user_input)
    if st.button("Parse Background"):
        result = parse_background(input_text)
        st.session_state.user_input = input_text
        st.session_state.structured_profile = result
        st.success("Profile Parsed:")
        st.success("Completed check the next page for more")

# --- TAB 2: Skill Mapping ---
with tabs[1]:
    st.subheader("Skill Mapping Agent")
    if st.session_state.structured_profile:
        skills = map_skills(st.session_state.structured_profile)
        st.markdown(skills)
    else:
        st.info("Please parse background first in Tab 1.")

# --- TAB 3: Unlearning Advisor ---
with tabs[2]:
    st.subheader("Unlearning Advisor Agent")
    domain = st.text_input("What domain are you pivoting to?")
    if st.session_state.structured_profile and domain:
        unlearn = unlearning_advice(st.session_state.structured_profile, domain)
        st.markdown(unlearn)
    else:
        st.info("Please complete Tab 1 and enter a target domain.")

# --- TAB 4: Track Recommender (RAG) ---
with tabs[3]:
    st.subheader("Track Recommender Agent (RAG)")
    query = st.text_input("Enter your career goal or domain interest:")
    if st.button("Recommend Learning Tracks"):
        if query:
            output = recommend_tracks(query)
            st.markdown(output)
        else:
            st.warning("Please enter a goal (e.g., 'Switch to cybersecurity').")
