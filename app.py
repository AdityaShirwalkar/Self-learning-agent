"""Private Streamlit frontend for the self-learning agent."""

import hmac

import streamlit as st

from config import APP_PASSWORD
from memory_agent import SelfLearningAgent

st.set_page_config(page_title="Self-Learning AI Agent", page_icon="🧠", layout="centered")


def require_access() -> None:
    """Protect a personal deployment with one shared password."""
    if not APP_PASSWORD:
        st.error("APP_PASSWORD is not configured. Add it to .env or deployment secrets before using the web app.")
        st.stop()
    if st.session_state.get("authenticated"):
        return

    st.title("🧠 Self-Learning AI Agent")
    password = st.text_input("App password", type="password")
    if st.button("Unlock", type="primary"):
        if hmac.compare_digest(password, APP_PASSWORD):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def reset_user_session(user_id: str) -> None:
    st.session_state.agent = SelfLearningAgent(user_id=user_id)
    st.session_state.user_id = user_id
    st.session_state.messages = []


require_access()
st.title("🧠 Self-Learning AI Agent")
st.caption("Your conversations are stored as searchable memories for this user ID.")

with st.sidebar:
    st.header("Session")
    user_id = st.text_input("User ID", value=st.session_state.get("user_id", "default_user"))
    st.caption("Use 1-64 letters, numbers, hyphens, or underscores.")
    if st.button("Switch user"):
        try:
            reset_user_session(user_id)
            st.rerun()
        except (RuntimeError, ValueError) as error:
            st.error(str(error))

    if st.session_state.get("user_id"):
        if st.button("Show stored memories"):
            try:
                memories = st.session_state.agent.get_all_memories()
                if memories:
                    for memory in memories:
                        st.write(f"- {memory['memory']}")
                else:
                    st.info("No memories stored yet.")
            except Exception as error:
                st.error(f"Could not load memories: {error}")

        confirm_delete = st.checkbox("I understand this permanently deletes this user's memories.")
        if st.button("Forget all memories", disabled=not confirm_delete):
            try:
                st.session_state.agent.forget_everything()
                st.session_state.messages = []
                st.success("Memories deleted.")
            except Exception as error:
                st.error(f"Could not delete memories: {error}")

if "agent" not in st.session_state:
    try:
        reset_user_session("default_user")
    except Exception as error:
        st.error(f"The agent could not start: {error}")
        st.stop()

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Say something...")
if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = st.session_state.agent.chat(prompt)
                st.markdown(reply)
                st.session_state.messages.append(("assistant", reply))
            except Exception as error:
                st.error(f"The request failed: {error}")
