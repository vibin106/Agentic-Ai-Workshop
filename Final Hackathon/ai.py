# Updated main.py with fixed RAG agents and proper UI agent flow

import os
import re
import json
import streamlit as st
from markdown import markdown
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferMemory

if "doc_result" not in st.session_state:
    st.session_state["doc_result"] = None

# Load .env and API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY not found in environment."

shared_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_llm(max_tokens=800):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_output_tokens=max_tokens,
        google_api_key=GOOGLE_API_KEY,
        verbose=True
    )

def clean_markdown(text):
    if isinstance(text, dict):
        text = json.dumps(text, indent=2)
    if hasattr(text, 'content'):
        text = text.content
    text = re.sub(r'\\1', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\_(.*?)\_', r'<i>\1</i>', text)
    return markdown(str(text))


def get_retriever_from_urls(urls):
    loader = WebBaseLoader(urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": -1.0, "k": 4}
)


def create_design_tool():
    retriever = get_retriever_from_urls([
        "https://uxdesign.cc/tagged/case-study",  # Rich multi-domain UX case studies
        "https://uxplanet.org/tagged/ux-case-study",  # App design walkthroughs
        "https://www.smashingmagazine.com/category/uxdesign",  # UX articles, standards
        "https://material.io/design",  # Official Google Material Guidelines
        "https://uxfol.io/blog/ux-case-studies",  # Full-cycle product design studies
        "https://www.behance.net/search/projects?search=ui%20ux%20case%20study"  # UI/UX inspiration across domains
    ])
    chain = RetrievalQA.from_chain_type(llm=get_llm(1000), retriever=retriever, return_source_documents=True)
    def run(input):
        prompt = f"""
You are a UX research assistant specialized in design pattern extraction. Analyze existing mobile applications related to the following:

- Product Type: '{input['project']}'
- Domain: '{input['domain']}'
- Primary User Activity: '{input['activity']}'
- Target Audience: (assume general unless specified)

Your task is to retrieve and synthesize insights from external design sources (Behance, Dribbble, Material Design Guidelines, UX blogs, etc.) using Retrieval-Augmented Generation (RAG).

### OUTPUT FORMAT:

1. **Top Reference Applications**  
List 3–5 mobile/web applications similar to the target product.  
- Mention each app's name and platform.  
- Include what they are known for in UX or functionality.

2. **Key UX Features**  
Summarize notable interaction and navigation patterns:  
- Layout structure (e.g., tab bars, cards, full-page modals)  
- Component placement (CTAs, headers, icons, etc.)  
- Interaction techniques (swipe, tap, micro-interactions)  
- Visual style highlights (typography, color schemes)

3. **Ergonomic & Accessibility Highlights**  
Point out accessibility and usability principles applied:  
- Font/readability  
- Gesture optimization  
- Color contrast  
- Support for assistive tools

### ADDITIONAL INSTRUCTIONS:
- Use only insights retrieved from the RAG corpus.  
- Provide reference source links (where applicable).  
- Avoid generic responses or hallucinated app names.  
- Keep the full response under 700 tokens.  
"""
        result = chain.invoke({"query": prompt})
        return result.get("result", str(result))
    return Tool(name="DesignTool", func=run, description="Extracts design patterns and inspiration using RAG.")

def create_journey_tool():
    retriever = get_retriever_from_urls([
        "https://www.behance.net/gallery/145239404/Mindful-Wellbeing-App-UX-Case-Study",
        "https://uxdesign.cc/user-flow-mapping-guide-32c6b9c5b9e6",
        "https://m2.material.io/design/navigation/understanding-navigation.html"
    ])
    chain = RetrievalQA.from_chain_type(llm=get_llm(800), retriever=retriever)
    def run(input):
        prompt = f"""
You are a UX journey mapping assistant. Based on the following information:

- Product Name: '{project}'
- Domain: '{domain}'
- Primary User Activity: '{activity}'
- Design inspirations and references from similar apps: (assume drawn from Agent 1)

Your task is to create a structured user journey flow that reflects real-world usage and industry-standard UI/UX design practices.

### OUTPUT FORMAT (in valid JSON):

Each step should include:
- step_number: Sequential number or nested number for branches (e.g., 2.1)
- screen: The name or purpose of the screen
- role: What this screen accomplishes (e.g., show list, confirm details)
- action: User interaction (e.g., tap button, swipe, fill form)

You must include:
1. A **main user flow** of 7–12 steps representing the core user activity
2. At least one **alternative branch** (e.g., error condition, fallback screen, user choice)
3. All content in a **clean JSON schema**, suitable for visualization

### EXAMPLE (partial):

```json
[
  {{
    "step_number": 1,
    "screen": "Home",
    "role": "Greet user and show main CTA",
    "action": "Tap 'Book Session'"
  }},
  {{
    "step_number": 2,
    "screen": "Session Catalog",
    "role": "Show list of breathing sessions",
    "action": "Select a session"
  }},
  {{
    "branch": {{
      "condition": "User not logged in",
      "step_number": "2.1",
      "screen": "Login",
      "role": "Request authentication",
      "action": "Enter credentials"
    }}
  }}
]
"""
        response = chain.run(prompt)
        try:
            return json.dumps(json.loads(response), indent=2)
        except:
            return response
    return Tool(name="JourneyTool", func=run, description="Builds a user journey using RAG and outputs structured flow.")

def create_screen_tool():
    llm = get_llm(800)
    def run(input):
        prompt = f"""
You are a UI/UX screen descriptor for mobile applications. Based on the structured user journey below, provide a per-screen breakdown to guide handoff to UI designers or no-code developers.

## CONTEXT:
- App Name: '{input['project']}'
- Domain: '{input['domain']}'
- Primary Activity: '{input['activity']}'
- Journey Flow (JSON):
{input['flow_schema']}

## YOUR TASK:
For each screen in the journey, describe:

1. **Screen Name**  
   Provide a clear label for the screen (e.g., "Booking Form").

2. **Layout Plan**  
   Textual description of layout zones.  
   Example:  
   - Top: App bar with back button and title  
   - Middle: Session cards in scrollable list  
   - Bottom: Primary CTA button

3. **Main UI Components**  
   List every visual or interactive element with role.  
   Example:  
   - Dropdown (select date)  
   - Button (confirm booking)  
   - Form field (enter email)

4. **Screen Logic**  
   Describe what happens on user interaction.  
   Example:  
   - On submit → navigate to confirmation screen  
   - If form invalid → show error message inline

5. **User Goals**  
   What is the user's intention or action on this screen?

6. **Accessibility Notes**  
   Include at least one UX accessibility tip.  
   Example:  
   - “Ensure 4.5:1 contrast for text”  
   - “Swipe gesture for tab navigation”  
   - “Label all icons with `aria-label` for screen readers”

## OUTPUT FORMAT:
Return each screen as a **numbered list** with sections for layout, logic, components, user goals, and accessibility notes. Do **not** output code or JSON — only clear, structured English.

## IMPORTANT:
- Be precise and professional — this will be used by designers to construct wireframes or Figma layouts.  
- Do not hallucinate data. Only describe what is implied by the journey flow.  
- Focus on usability, visual hierarchy, and clarity of interaction.
"""
        return llm.invoke(prompt)
    return Tool(name="ScreenTool", func=run, description="Describes screen layout, logic, and accessibility.")

def create_storyboard_tool():
    llm = get_llm(800)
    def run(input):
        prompt = f"""
You are a storyboard generation assistant. Based on the following app context and screen definitions, create a textual storyboard outlining all screen-to-screen transitions, user triggers, and state changes.

## CONTEXT:
- App Name: '{input['project']}'
- Domain: '{input['domain']}'
- Primary Activity: '{input['activity']}'
- Screen Descriptions:
{input['screen_descriptions']}

## TASK:
Create a **text-based storyboard** that describes transitions between screens. Include:

1. **From Screen → To Screen**  
   Use this format:  
   `Home → taps 'Start' → navigates (slide) → Session List`

2. **User Triggers**  
   Indicate what action causes the transition (e.g., tap, swipe, auto-timer).

3. **Transition Style**  
   Describe the navigation pattern (e.g., fade, modal, slide, overlay).

4. **State Changes**  
   Note if the app changes internally (e.g., login state, data selected, error state).

5. **Conditional Branches**  
   Include at least one alternate flow or edge case.  
   Example:  
   `Booking → submits with empty form → modal error → remains on Booking`

## FORMAT:
Return a **numbered list** of transition descriptions like a script.  
Each transition line should include:

- Screen Name (From)
- Trigger (Action or condition)
- Screen Name (To)
- Transition style
- Any relevant state change or branch info

## ADDITIONAL INSTRUCTIONS:
- Include **all steps and branches** from the user journey.
- Reflect realistic UX behavior and standard navigation flows.
- Use clear, structured, and human-readable wording — this will guide prototyping and animation.

## EXAMPLE (Do not copy directly):
1. Home → taps “Book Now” → slide transition → Session Selection  
2. Session Selection → selects a session → fade transition → Booking Form  
3. Booking Form → submits incomplete form → modal alert → remains on Booking Form  
4. Booking Form → valid submit → slide transition → Confirmation
"""
        return llm.invoke(prompt)
    return Tool(name="StoryboardTool", func=run, description="Outlines transitions between UI screens.")

def create_doc_tool():
    retriever = get_retriever_from_urls([
        "https://uxdesign.cc/documentation-design-decisions-ux-case-study-bcf56662e1f3",
        "https://www.behance.net/search/projects?search=mobile+app+case+study",
        "https://m2.material.io/design/guidelines-overview"
    ])
    # Custom LLM with 2000 tokens just for Agent 5
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_output_tokens=6000,  # ← Set max tokens here
        google_api_key=GOOGLE_API_KEY,
        verbose=True
    )
    chain = RetrievalQA.from_chain_type(llm=get_llm(1000), retriever=retriever)
    def run(input):
        prompt = f"""
You are a professional UX documentation assistant. Your task is to compile a complete and developer-ready **UI/UX design specification document** for the following application.

## CONTEXT:
- App Name: '{input["project"]}'
- Domain: '{input["domain"]}'
- Primary Activity: '{input["activity"]}'
- Design Inspirations:  
{input["inspirations"]}
- Journey Flow (JSON):  
{input["flow_schema"]}
- Screen Descriptions:  
{input["screen_descriptions"]}
- Storyboard:  
{input["storyboard"]}

## OUTPUT:

Create a structured design spec in clean Markdown format (Notion-compatible) with the following sections.It should be structured in clean Markdown format, suitable for Notion or PDF export. Include the following sections:
in a professional tone. The output should be limited to a total of 6000 tokens.

---

### 🧩 1. Project Overview
- Brief description of the product, domain, and user intent.

---

### 💡 2. UI/UX Rationale (Per Screen)
For each screen:
- Screen Name  
- Design decisions made  
- Justification in terms of UX best practices (e.g., simplicity, accessibility, clarity)  
- How it supports the user journey and app goals

---

### 🎨 3. Design Inspirations (From Retrieved Sources)
- List 2–3 relevant UI references or similar apps  
- Include links to Behance, Dribbble, Material.io, etc.  
- Describe the takeaway from each inspiration and how it influenced the prototype

---

### 🧭 4. Journey Flow (Visual Logic)
- Explain the flow in short text: entry → process → goal  
- Highlight any branches or conditional paths

---

### 📱 5. Screen Descriptions
- Recap each screen’s components, layout, and logic  
- Mention any accessibility or ergonomic notes

---

### 🎬 6. Storyboard Transitions
- Textual storyboard format showing how users move between screens  
- Include action → transition → destination steps  
- Mention conditional or error-based transitions

---

### ✅ 7. Developer Checklist
- [ ] High contrast for text and buttons  
- [ ] Logical tab order / keyboard support  
- [ ] Aria labels on all icons and interactive elements  
- [ ] Responsive layout tested across screen sizes  
- [ ] Form error handling and validation messages

---

### 🔗 8. Reference Links
- Material Design: https://m2.material.io  
- Heuristic Design Principles: https://www.nngroup.com/articles/ten-usability-heuristics/  
- Retrieved UX sources (insert source links here)

---

## REQUIREMENTS:
- Be concise and professional — this document will be exported as PDF or published to Notion.
- Tie all design choices back to known UX heuristics, standards, or source links.
- Do not hallucinate references — use RAG-retrieved data only where available.
"""
        return chain.run(prompt)
    return Tool(name="DocumentationTool", func=run, description="Compiles UX spec using all prior outputs.")

# Agent and Tools

design_tool = create_design_tool()
journey_tool = create_journey_tool()
screen_tool = create_screen_tool()
storyboard_tool = create_storyboard_tool()
doc_tool = create_doc_tool()

agent = initialize_agent(
    tools=[design_tool, journey_tool, screen_tool, storyboard_tool, doc_tool],
    llm=get_llm(1000),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=shared_memory,
    verbose=True  
)

# Streamlit Interface

st.set_page_config(page_title="Agentic RAG UI/UX AI", layout="wide")
st.title("🎯 AI-Powered UI/UX Prototype Generator with RAG Agents")

with st.form("input_form"):
    project = st.text_input("Project Name")
    domain = st.text_input("Domain")
    activity = st.text_input("Primary Activity")
    submitted = st.form_submit_button("Generate")

if submitted and project and domain and activity:
    input_data = {"project": project, "domain": domain, "activity": activity}

    with st.spinner("🔍 Running Design Agent..."):
        design_result = design_tool.func(input_data)
        st.subheader("🎨 Design Insights")
        st.markdown(clean_markdown(design_result), unsafe_allow_html=True)

    with st.spinner("🔍 Generating Journey Map..."):
        journey_result = journey_tool.func(input_data)
        input_data["flow_schema"] = journey_result
        st.subheader("🧭 User Journey")
        st.code(journey_result, language="json")

    with st.spinner("🔍 Creating Screen Layouts..."):
        screen_result = screen_tool.func(input_data)
        input_data["screen_descriptions"] = screen_result
        st.subheader("📱 Screens")
        st.markdown(clean_markdown(screen_result), unsafe_allow_html=True)

    with st.spinner("🔍 Storyboarding..."):
        storyboard_result = storyboard_tool.func(input_data)
        input_data["storyboard"] = storyboard_result
        st.subheader("🎬 Storyboard")
        st.markdown(clean_markdown(storyboard_result), unsafe_allow_html=True)

    with st.spinner("📘 Final Documentation..."):
        input_data["inspirations"] = design_result  # ← Fix is here
        st.session_state["doc_result"] = doc_tool.func(input_data)
        st.subheader("📘 Prototype Specification")
        if st.session_state["doc_result"]:
            st.markdown(clean_markdown(st.session_state["doc_result"]), unsafe_allow_html=True)
            cleaned_doc = re.sub(r'\*+', '', st.session_state["doc_result"])
            st.download_button(
                "📥 Download Final Document",
                cleaned_doc,
                "final_documentation.txt"
            )



