import streamlit as st
from google import genai
from google.genai import types

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AI Assistant", page_icon="✨", layout="wide")

# Custom Minimalist CSS
st.markdown("""
    <style>
    .stApp { background-color: #fdfdfd; }
    .stChatMessage { 
        padding: 1.5rem; 
        border-radius: 12px; 
        margin-bottom: 1rem; 
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stChatInputContainer { padding-bottom: 2rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #eee; }
    </style>
""", unsafe_allow_html=True)

# --- 2. INITIALIZATION ---
# Securely get API Key from Streamlit Secrets
# (You will set this key in the Streamlit Cloud dashboard later)
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Model Configuration
with st.sidebar:
    st.title("Settings")
    model_choice = st.selectbox("Select Model", ["gemini-2.0-flash", "gemini-2.0-pro-exp-02-05"])
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("2026 Unified AI Interface")

# --- 3. CHAT LOGIC ---
st.title("Personal AI Assistant")
st.write("A clean, private workspace for generic prompting.")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Type your message here..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Actual AI Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Call Gemini 3
        response = client.models.generate_content(
            model=model_choice,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temp)
        )
        
        full_response = response.text
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})