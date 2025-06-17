import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import GEMINI_MODEL

# ────── UI Setup ──────
st.set_page_config(page_title="Unlearning Advisor Agent", layout="centered")
st.title("🧠 Unlearning Advisor Agent")
st.markdown("Get advice on what **mindsets, habits, or tools** to unlearn when switching careers.")

user_background = st.text_area(
    "✍️ Describe your background (roles, habits, tools used, etc.):",
    placeholder="Example: I worked in finance for 6 years, used Excel daily, focused on risk-averse decision-making and documentation-heavy workflows."
)

target_domain = st.text_input(
    "🎯 Target career domain (optional but recommended):",
    placeholder="e.g., Data Science, Cloud Engineering, Product Design"
)

# ────── Prompt Setup ──────
prompt = PromptTemplate.from_template("""
You are an AI Unlearning Advisor. Based on the user's background and target career domain, suggest what they should consciously unlearn to succeed in the transition.

Background:
{background}

Target Domain:
{domain}

Output 5–7 bullet points. For each, include:
- What to unlearn
- Why it's important to let go of it
""")

chain = (
    prompt
    | ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.5, max_output_tokens=700)
    | StrOutputParser()
)

# ────── Agent Run ──────
if user_background:
    with st.spinner("Analyzing unlearning points..."):
        result = chain.invoke({
            "background": user_background,
            "domain": target_domain or "a new field"
        })
        st.success("🧠 Recommended Things to Unlearn:")
        st.markdown(result)
