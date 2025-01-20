import sys
sys.path.append(".")
import streamlit as st
from datetime import datetime
class ChatPage:
    def __init__(self, app):
        """Initialize with a reference to the main app."""
        self.app = app

    def render_chat_page(self):
        """Render the chat interface."""
        if not st.session_state.get("current_session_id"):
            st.error("No active chat session found. Please create a new chat.")
            if st.button("⬅️ Back to Home"):
                st.session_state.active_page = "home"
                st.session_state.layout = "centered"
                st.rerun()
            return
        chat_name = st.session_state.sessions[st.session_state.current_session_id]["name"]
        st.markdown(f"<h1 style='text-align: left;'>Chat with H1 : {chat_name} </h1>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Home"):
            st.session_state.active_page = "home"  # Go back to home page
            st.session_state.layout = "centered"
            st.session_state.sessions[st.session_state.current_session_id]["last_saved_index"] = -1
            st.rerun()
        # Display the chat messages and handle input
        self.app.display_chat()


    def render_pdf_chat_page(self):
        if not st.session_state.get("current_session_id"):
            st.error("No active chat session found. Please create a new chat.")
            if st.button("⬅️ Back to Home"):
                st.session_state.active_page = "home"
                st.session_state.layout = "centered"
                st.rerun()
            return
        chat_name = st.session_state.sessions[st.session_state.current_session_id]["name"]
        st.markdown(f"<h1 style='text-align: left;'>PDF Chat with H1 : {chat_name} </h1>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Home"):
            st.session_state.active_page = "home"  # Go back to home page
            st.session_state.layout = "centered"
            st.rerun()

        self.app.display_chat()

    def render_machine_learning_page(self):
        pass
    


    