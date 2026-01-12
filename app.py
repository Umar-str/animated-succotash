import streamlit as st
import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader # New library for PDF parsing

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="Football Transfer RAG", page_icon="⚽")

# --- 2. INITIALIZE CLIENTS ---
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# --- 3. PARSING LOGIC ---
def parse_document(uploaded_file):
    """Extracts text from PDF or TXT files."""
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()
    elif uploaded_file.type == "text/plain":
        text = str(uploaded_file.read(), "utf-8")
    return text

@st.cache_resource
def get_vector_db():
    chroma_client = chromadb.EphemeralClient()
    return chroma_client.get_or_create_collection(name="football_transfers")

collection = get_vector_db()

# --- 4. SIDEBAR UPLOADER ---
with st.sidebar:
    st.title("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload Transfer News (PDF/TXT)", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Analyzing and Indexing..."):
            raw_text = parse_document(uploaded_file)
            # Simple chunking: split by double newlines or large blocks
            chunks = [c.strip() for c in raw_text.split('\n\n') if len(c.strip()) > 10]
            
            for i, chunk in enumerate(chunks):
                # Using Gemini to embed each chunk
                resp = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=chunk,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                collection.add(
                    ids=[f"{uploaded_file.name}_{i}"],
                    embeddings=[resp.embeddings[0].values],
                    documents=[chunk]
                )
        st.success(f"Indexed {len(chunks)} sections from {uploaded_file.name}!")

# --- 5. CHAT INTERFACE ---
st.title("Perk HRMS Expert")
st.caption("Gemini 3 Powered RAG System")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about company policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute RAG Logic
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Retrieve
            context = get_relevant_context(prompt)
            
            # 2. Generate with Gemini 3
            system_msg = "You are a Perk HRMS expert. Answer ONLY using the provided context."
            user_prompt = f"CONTEXT: {context}\n\nQUESTION: {prompt}"
            
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_msg,
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )
            
            full_response = response.text
            st.markdown(full_response)
            
            # Optional: Show reasoning in an expander
            with st.expander("View AI Reasoning"):
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        st.caption(part.text)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
