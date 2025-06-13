import streamlit as st
import PyPDF2
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from dotenv import load_dotenv

# Optional: Load from .env file
load_dotenv()

# Set your Gemini API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or "AIzaSyAECDPLU_cIAY7sk12aq8__G5CkoEQGqTY"

# Initialize Gemini model via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# PDF text extraction function
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Define the summary prompt and chain
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
Summarize the following study material into concise bullet points (3 to 5 points):

Study Material:
{content}

Summary:
"""
)
summary_chain: Runnable = summary_prompt | llm | StrOutputParser()

# Define the quiz generation prompt and chain
quiz_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
Based on the following summary, generate 2 multiple-choice quiz questions. Each question should have 4 options, and the correct answer should be clearly marked.

Summary:
{summary}

Format:
1. Question?
   a) Option 1
   b) Option 2
   c) Option 3
   d) Option 4
Answer: <correct option>

2. ...
"""
)
quiz_chain: Runnable = quiz_prompt | llm | StrOutputParser()

# Streamlit UI
st.title("📘 Study Assistant using LangChain + Gemini")
st.write("Upload a course document (PDF) to generate summary and MCQ quiz questions.")

uploaded_file = st.file_uploader("Upload your study material (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("📄 Extracting text from PDF..."):
        study_text = extract_text_from_pdf(uploaded_file)

    if study_text:
        with st.spinner("✍️ Summarizing content..."):
            summary = summary_chain.invoke({"content": study_text})

        with st.spinner("🧠 Generating quiz questions..."):
            quiz = quiz_chain.invoke({"summary": summary})

        st.subheader("✅ Summary")
        st.markdown(summary)

    st.subheader("📝 Quiz Questions")
    st.markdown(quiz)
    # Combine output into a single string
    combined_output = f"Summary:\n{summary}\n\nQuiz Questions:\n{quiz}"

    # Download button
    st.download_button(
        label="📥 Download Summary & Quiz",
        data=combined_output,
        file_name="study_summary_and_quiz.txt",
        mime="text/plain"
    )

else:
    st.error("❌ No readable text found in the uploaded PDF.")
