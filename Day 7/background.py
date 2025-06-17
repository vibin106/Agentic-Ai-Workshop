import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import GEMINI_MODEL

# ────── UI Setup ──────
st.set_page_config(page_title="Background Parsing Agent", layout="centered")
st.title("🧾 Background Parsing Agent")
st.markdown("Convert your **raw career background** into a structured profile.")

user_input = st.text_area(
    "✍️ Enter your background details (education, work, interests):",
    placeholder="Example: I studied electrical engineering, worked in embedded systems for 5 years, and now want to move into cloud computing."
)

# ────── LLM Setup ──────
prompt = PromptTemplate.from_template("""
You are an AI assistant that extracts a structured professional profile from user text.

Text: {user_background}

Output the result as a clear JSON object with keys:
- Education
- Previous Roles
- Years of Experience
- Career Interest
""")

chain = (
    prompt
    | ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, max_output_tokens=512)
    | StrOutputParser()
)

# ────── Agent Run ──────
if user_input:
    with st.spinner("Parsing your background..."):
        profile = chain.invoke({"user_background": user_input})
        st.success("✅ Structured Profile Generated:")
        st.code(profile, language="json")
