import streamlit as st
import PyPDF2
import networkx as nx
import matplotlib.pyplot as plt

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

# ✅ Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    api_key="AIzaSyAKDDAPl3nWnVg3WzrEKlRDEOeA5-ZK92Q"  # Replace with your actual key
)

# ✅ Prompt template
prompt = PromptTemplate.from_template("""
You are a Career Transition Assistant.

User Background:
- Education: {education}
- Previous Roles: {roles}
- Interests: {interests}

Extracted Knowledge (from PDF): {rag_context}

Based on this, list:
1. Transferable Skills
2. Obsolete Skills
3. Suggested Learning Tracks
4. Suitable Roles
5. What to unlearn

Answer in simple bullet points.
""")

# ✅ Chain setup
chain: Runnable = prompt | llm | StrOutputParser()

# ✅ Extract PDF text
def extract_pdf_text(file, max_chars=3000):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if len(text) > max_chars:
            break
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text[:max_chars]

# ✅ Parse Gemini output to structured dictionary
def parse_gemini_output(output_text):
    sections = {
        "Transferable Skills": [],
        "Obsolete Skills": [],
        "Suggested Learning Tracks": [],
        "Suitable Roles": [],
        "What to unlearn": []
    }
    current = None
    for line in output_text.splitlines():
        line = line.strip()
        for key in sections.keys():
            if line.startswith(f"• {key}"):
                current = key
                break
        if current and line.startswith("◦"):
            sections[current].append(line.replace("◦", "").strip())
    return sections

# ✅ Generate knowledge graph
def generate_graph(output_text):
    parsed = parse_gemini_output(output_text)
    G = nx.DiGraph()
    G.add_node("User Background")
    for category, items in parsed.items():
        G.add_node(category)
        G.add_edge("User Background", category)
        for item in items:
            G.add_node(item)
            G.add_edge(category, item)
    return G

# ✅ Draw graph using Matplotlib
def draw_graph(G):
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray",
            node_size=2000, font_size=9, font_weight='bold', ax=ax)
    return fig

# ✅ Streamlit UI
st.set_page_config(page_title="Cross-Domain AI Navigator")
st.title("🔍 Cross-Domain AI Navigator")
st.markdown("Upload a career-related PDF and describe your background to discover your next move.")

education = st.text_input("🎓 Educational Background")
roles = st.text_input("💼 Previous Roles")
interests = st.text_input("🎯 Career Interests")

if st.button("🔍 Analyze"):
    with st.spinner("Extracting and Analyzing..."):
        try:
            with open("Career_Domain_Knowledge_Guide.pdf", "rb") as file:
                pdf_text = extract_pdf_text(file)

            output = chain.invoke({
                "education": education,
                "roles": roles,
                "interests": interests,
                "rag_context": pdf_text
            })

            # 🧠 Show suggestions once
            st.subheader("🧠 Career Mapping Suggestions")
            st.markdown(output)

            # 📄 Download
            st.download_button(
                label="📄 Download Suggestions",
                data=output,
                file_name="career_mapping_suggestions.txt",
                mime="text/plain"
            )

            # 🌐 Knowledge Graph
            st.subheader("🌐 Knowledge Graph View")
            G = generate_graph(output)
            fig = draw_graph(G)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
