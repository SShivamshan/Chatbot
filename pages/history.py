import sys
sys.path.append(".")
import streamlit as st
from datetime import datetime

"""
Goes to the history page containing old chats to be removed or to view it's contents.
"""
class HistoryPage:
    def __init__(self, app):
        """Initialize with a reference to the main app."""
        self.app = app
        # st.title("History")

    def render_history_page(self):
        """Render the chat history interface."""
        ai = None
        user = None
        st.markdown("<h1 style='text-align: left;'>Chat History</h1>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Home"):
           st.session_state.active_page = "home"
           st.session_state.layout = "centered"
           st.rerun()

        # Check if chat history exists
        if "sessions" in st.session_state and st.session_state.sessions:
            st.markdown("### Select a session to view its history:")
            for session_id, session_data in st.session_state.sessions.items():
                with st.expander(f"📂 {session_data['name']}"):
                    # Display each message with better formatting
                    for msg in session_data["messages"]:
                        if "User" in msg:
                            user = msg['User']
                        elif "AI" in msg:
                            ai = msg['AI']
                            st.markdown(f"**User**: {user} \t **AI**: {ai}")

                    # Delete button inside the expander
                    delete_button_placeholder = st.empty()
                    if delete_button_placeholder.button("❌ Delete Session", key=f"delete_{session_id}"):
                        # print(f"Delete button clicked for session ID: {session_id}")
                        self.app.delete_session(session_id)
                        st.success(f"Session '{session_data['name']}' deleted.")
                        st.rerun()
                
        else:
            st.write("No chat history available.")