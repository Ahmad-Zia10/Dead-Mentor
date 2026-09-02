import streamlit as st
import requests

# --- Config ---
API_URL = "http://127.0.0.1:8000"

MENTOR_DISPLAY = {
    "marcus": "Marcus Aurelius — Stoic Emperor",
    "feynman": "Richard Feynman — Physicist",
    "darwin": "Charles Darwin — Naturalist"
}

# --- Page Config ---
st.set_page_config(
    page_title="Dead Mentor",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Dead Mentor")
st.caption("Converse with history's greatest minds")

# --- Mentor Selection ---
mentor_key = st.selectbox(
    "Choose your mentor",
    options=list(MENTOR_DISPLAY.keys()),
    format_func=lambda x: MENTOR_DISPLAY[x]
)

st.divider()

# --- Session State Init ---
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_mentor" not in st.session_state:
    st.session_state.current_mentor = mentor_key

# --- Reset conversation when mentor changes ---
if st.session_state.current_mentor != mentor_key:
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state.current_mentor = mentor_key

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📜 Sources"):
                for source in msg["sources"]:
                    st.write(f"- {source}")

# --- Chat Input ---
if query := st.chat_input(f"Ask {MENTOR_DISPLAY[mentor_key].split('—')[0].strip()} anything..."):

    # Display user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Call API
    with st.spinner("Consulting the archives..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "mentor": mentor_key,
                    "query": query,
                    "history": st.session_state.history
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            answer = data["response"]
            sources = data["sources"]
            st.session_state.history = data["history"]

            # Display mentor response
            with st.chat_message("assistant"):
                st.write(answer)
                with st.expander("📜 Sources"):
                    for source in set(sources):
                        st.write(f"- {source}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": list(set(sources))
            })

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# --- Clear conversation button ---
if st.button("🗑️ Clear Conversation"):
    st.session_state.history = []
    st.session_state.messages = []
    st.rerun()