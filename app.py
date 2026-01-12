import streamlit as st
from google import genai
from google.genai import types

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Gemini 3 Assistant", page_icon="✨", layout="wide")

# --- 2. INITIALIZATION ---
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Gemini 3 Settings
with st.sidebar:
    st.title("Settings")
    # Updated to Gemini 3 Model IDs
    model_choice = st.selectbox("Select Model", [
        "gemini-3-flash-preview", 
        "gemini-3-pro-preview"
    ])
    
    # Gemini 3 specific: Thinking Level (Low for speed, High for complex reasoning)
    thinking_level = st.select_slider(
        "Thinking Level", 
        options=["minimal", "low", "medium", "high"], 
        value="low"
    )
    
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- 3. CHAT LOGIC ---
st.title("Gemini 3 Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Call Gemini 3 with new Thinking Config
        response = client.models.generate_content(
            model=model_choice,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temp,
                thinking_config=types.ThinkingConfig(
                    thinking_level=thinking_level
                )
            )
        )
        
        full_response = response.text
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})