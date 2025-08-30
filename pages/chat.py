import sys
sys.path.append(".")
import streamlit as st
from datetime import datetime

class ChatPage:
    def __init__(self, app):
        """
        Initialize with a reference to the main app.

        params
        ------ 
        - app: The main ChatbotApp instance.
        """
        self.app = app

    def render_chat_page(self):
        """
        Render the chat interface.
        """
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
            chat_type = st.session_state.sessions[st.session_state.current_session_id].get("chat_type", None)
            if st.session_state.llm_instances[chat_type] is not None:
                if hasattr(st.session_state.llm_instances[chat_type].llm, "unload_model") and callable(getattr(st.session_state.llm_instances[chat_type].llm, "unload_model")):
                    st.session_state.llm_instances[chat_type].llm.unload_model()
            st.rerun()
        # Display the chat messages and handle input
        self.app.display_chat()


    def render_pdf_chat_page(self):
        """
        Render the chat page.
        """
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
            chat_type = st.session_state.sessions[st.session_state.current_session_id].get("chat_type",None)
            if st.session_state.llm_instances.get(chat_type,None):
                if len(st.session_state.llm_instances[chat_type]) > 0:
                    if hasattr(st.session_state.llm_instances[chat_type].llm, "unload_model") and callable(getattr(st.session_state.llm_instances[chat_type].llm, "unload_model")):
                        st.session_state.llm_instances[chat_type][st.session_state.current_session_id].llm.unload_model()
            st.rerun()

        self.app.display_chat()

    def render_AGENT_page(self):
        """
        Render the AGENT page.
        """
        if not st.session_state.get("current_session_id"):
            st.error("No active chat session found. Please create a new chat.")
            if st.button("⬅️ Back to Home"):
                st.session_state.active_page = "home"
                st.session_state.layout = "centered"
                st.rerun()
            return
        chat_name = st.session_state.sessions[st.session_state.current_session_id]["name"]
        st.markdown(f"<h1 style='text-align: left;'>Agent : {chat_name} </h1>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Home"):
            st.session_state.active_page = "home"  # Go back to home page
            st.session_state.layout = "centered"
            chat_type = st.session_state.sessions[st.session_state.current_session_id].get("chat_type",None)
            if st.session_state.llm_instances.get(chat_type,None):
                if len(st.session_state.llm_instances[chat_type]) > 0:
                    if hasattr(st.session_state.llm_instances[chat_type].llm, "unload_model") and callable(getattr(st.session_state.llm_instances[chat_type].llm, "unload_model")):
                        st.session_state.llm_instances[chat_type].llm.unload_model()
            st.rerun()

        self.app.display_chat()
        
    


    