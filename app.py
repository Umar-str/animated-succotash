import streamlit as st
import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="Football Transfer RAG", page_icon="⚽")

# --- 2. INITIALIZE CLIENTS ---
# Ensure GEMINI_API_KEY is set in Streamlit Cloud Secrets
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# --- 3. HELPER FUNCTIONS ---

def parse_document(uploaded_file):
    """Extracts text from PDF or TXT files."""
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif uploaded_file.type == "text/plain":
        text = str(uploaded_file.read(), "utf-8")
    return text

@st.cache_resource
def get_vector_db():
    """Initializes the ChromaDB collection once."""
    chroma_client = chromadb.EphemeralClient()
    return chroma_client.get_or_create_collection(name="football_transfers")

def get_relevant_context(query_text, _collection, num_results=2):
    """Retrieves relevant chunks from the vector database."""
    # 1. Embed the user query
    query_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vector = query_resp.embeddings[0].values

    # 2. Query the collection
    results = _collection.query(
        query_embeddings=[query_vector],
        n_results=num_results
    )
    
    # Handle empty results
    if not results or not results['documents'][0]:
        return "No relevant context found."
        
    return " ".join(results['documents'][0])

# --- 4. DATA INITIALIZATION ---
collection = get_vector_db()

# --- 5. SIDEBAR UPLOADER ---
with st.sidebar:
    st.title("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload Transfer News (PDF/TXT)", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Analyzing and Indexing..."):
            raw_text = parse_document(uploaded_file)
            chunks = [c.strip() for c in raw_text.split('\n\n') if len(c.strip()) > 10]
            
            for i, chunk in enumerate(chunks):
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
        st.success(f"Indexed {len(chunks)} sections!")

# --- 6. CHAT INTERFACE ---
st.title("Football Transfer Expert")
st.caption("Gemini 3 Powered RAG System (Sept 2025 - Jan 2026)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

system_msg = input("Write your system message.")

# User Input
if prompt := st.chat_input("Ask about 2026 transfers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute RAG Logic
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & thinking..."):
            # 1. Retrieve (Passing the collection explicitly)
            context = get_relevant_context(prompt, collection)
            
            # 2. Generate with Gemini 3
            #system_msg = input("Write your system message.")
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
            
            """Show reasoning
           with st.expander("View AI Reasoning"):
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        st.caption(part.text)"""
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})