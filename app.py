import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="Perk HRMS Assistant", page_icon="🏢", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    .stChatMessage { border-radius: 15px; border: 1px solid #eee; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. INITIALIZE CLIENTS ---
# Uses st.secrets for safety in the cloud
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# --- 3. CACHED VECTOR DB LOGIC ---
@st.cache_resource
def get_vector_db():
    """Builds the DB once and caches it for the whole session."""
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name="perk_temp_db")
    
    # Your Colab documents
    documents = [
        "The standard Work-from-Home (WFH) policy allows employees to work remotely up to two days per week with prior manager approval.",
        "Travel expenses exceeding $50 require a digital receipt for reimbursement, which must be submitted via the portal within 15 days.",
        "The annual leave cycle runs from January to December. Employees can carry forward a maximum of 5 unused days to the next calendar year.",
        "To reset your internal system password, visit the 'Security Settings' tab and select 'Update Credentials'. Passwords must be 12 characters long.",
        "Our internal servers undergo scheduled maintenance every first Sunday of the month between 02:00 AM and 04:00 AM EST.",
        "Hardware upgrades for laptops are available every 36 months of service. Requests should be logged under the 'IT Procurement' ticket category.",
        "The company headquarters is located in the downtown Innovation District, featuring an open-plan design and three dedicated collaborative zones.",
        "Our sustainability initiative aims to reduce office paper waste by 40% by the end of 2026 through the 'Digital-First' documentation mandate.",
        "The 'Peer Recognition' program allows team members to nominate colleagues for monthly awards based on core values like Integrity and Innovation.",
        "All employees must complete the Mandatory Cybersecurity Training module annually to maintain their system access privileges.",
        "Confidential documents should never be shared via external messaging apps; use the secure internal 'Vault' for all sensitive file transfers."
    ]

    # Indexing documents (happens only on first run)
    for i, text in enumerate(documents):
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        vector = response.embeddings[0].values
        collection.add(ids=[str(i)], embeddings=[vector], documents=[text])
    
    return collection

# Initialize the collection
collection = get_vector_db()

# --- 4. RAG HELPER FUNCTIONS ---
def get_relevant_context(query_text, num_results=2):
    query_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vector = query_resp.embeddings[0].values
    results = collection.query(query_embeddings=[query_vector], n_results=num_results)
    return " ".join(results['documents'][0])

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