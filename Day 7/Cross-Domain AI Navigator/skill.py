import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ─────── SETUP ───────
from config import GEMINI_MODEL, EMBEDDING_MODEL


# Simple domain skill reference
domain_skills = {
    "Cloud Engineering": [
        "Linux", "AWS", "GCP", "Azure", "CI/CD", "Docker", "Kubernetes", "Terraform", "Networking", "Scripting"
    ],
    "Data Science": [
        "Python", "Pandas", "Numpy", "Statistics", "Machine Learning", "SQL", "Data Visualization", "Scikit-learn"
    ],
    "UI/UX Design": [
        "Figma", "Prototyping", "User Research", "Interaction Design", "Adobe XD", "Usability Testing", "Design Thinking"
    ],
    "Cybersecurity": [
        "Network Security", "Linux", "Ethical Hacking", "Penetration Testing", "Firewalls", "SIEM", "Cryptography"
    ],
    # Add more as needed
}

# ─────── Streamlit UI ───────
st.set_page_config(page_title="Skill Mapping Agent", layout="wide")
st.title("🧠 Skill Mapping Agent")
st.markdown("Identify your **transferable, missing, and misaligned** skills for a new domain.")

user_skills = st.text_area("🔍 Enter your current skills (comma-separated):", placeholder="e.g., Python, Java, SQL, Excel")
target_domain = st.selectbox("🎯 Select a target domain:", list(domain_skills.keys()))

# ─────── Logic ───────
def generate_skill_mapping(user_input: str, domain: str):
    known = domain_skills[domain]
    prompt = PromptTemplate.from_template(
        """You are an expert in career guidance. The user has listed their current skills and wants to transition into {domain}.
Below are the domain's required skills:
{domain_skills}

Here are the user's skills:
{user_skills}

Analyze and categorize the user’s skills as:
1. ✅ Transferable
2. ⚠️ Missing (should be learned)
3. ❌ Misaligned or outdated

Respond in a clear, bullet-pointed format with reasoning."""
    )

    chain = (
        prompt
        | ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4, max_output_tokens=512)
        | StrOutputParser()
    )

    return chain.invoke({
        "domain": domain,
        "domain_skills": ", ".join(domain_skills[domain]),
        "user_skills": user_input
    })

# ─────── Result ───────
if user_skills and target_domain:
    with st.spinner("Analyzing skill gaps..."):
        result = generate_skill_mapping(user_skills, target_domain)
        st.success("🔍 Skill Mapping Result:")
        st.write(result)
