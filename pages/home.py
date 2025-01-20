import sys
sys.path.append(".")
import streamlit as st
from datetime import datetime

class HomePage:
    def __init__(self, app):
        """
        Initialize with a reference to the main app.
        :param app: The main ChatbotApp instance.
        """
        self.app = app
        

    def render_welcome_section(self,username:str):
        """Welcome Section."""
        greeting = self.get_greeting()
        st.markdown(f"<h1>🤖 Welcome to H1, {username} </h1>", unsafe_allow_html=True)
        st.markdown(f"<p>{greeting}! I'm H1, your AI assistant. How can I help you today?</p>", unsafe_allow_html=True)

    def get_greeting(self):
        """Get time-based greeting."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good Morning"
        elif 12 <= hour < 17:
            return "Good Afternoon"
        else:
            return "Good Evening"
        
    def render_features(self):
        """Render feature highlights in a horizontal layout using columns."""
        st.subheader("What I Can Do")

        # Add custom CSS for styling the feature cards
        st.markdown("""
            <style>
                .feature-card {
                    background-color: #ffffff; /* White card background */
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px; /* Adds spacing around cards */
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    text-align: center; /* Center-align text */
                }
                .feature-title {
                    font-size: 18px;
                    font-weight: bold;
                    color: black; /* Black text for title */
                    margin-bottom: 5px;
                }
                .feature-icon {
                    font-size: 30px;
                    margin-bottom: 10px;
                    display: block; /* Ensure icon and text stack */
                    color: black; /* Optional: Ensure icon matches text color */
                }
                .feature-description {
                    font-size: 14px;
                    color: black; /* Black text for description */
                }
            </style>
        """, unsafe_allow_html=True)

        # Features to display
        features = [
            {
                "icon": "💡",
                "title": "General Assistance",
                "description": "Ask me anything! From simple questions to complex problems."
            },
            {
                "icon": "📊",
                "title": "Data Analysis",
                "description": "Help with analyzing data, creating visualizations, and generating insights."
            },
            {
                "icon": "✍️",
                "title": "Writing Assistant",
                "description": "Help with writing, editing, and improving your content."
            }
        ]

        # Create columns for the features
        col1, col2, col3 = st.columns(3, vertical_alignment="center",gap="medium")

        # Assign features to columns
        cols = [col1, col2, col3]
        for col, feature in zip(cols, features):
            with col:
                st.markdown(f"""
                    <div class='feature-card'>
                        <span class='feature-icon'>{feature['icon']}</span>
                        <div class='feature-title'>{feature['title']}</div>
                        <div class='feature-description'>{feature['description']}</div>
                    </div>
                """, unsafe_allow_html=True)
        

    def render_home_page(self):
        """
        Render the home page with navigation options.
        Provides buttons for creating a new chat, viewing history, or accessing settings.
        """
        username = None
        if st.session_state.logged_in:
            username = self.app.account_page.username

        self.render_welcome_section(username=username)
        cols = st.columns(3)

        with cols[0]:
            options=("New Chat", "Machine learning", "PDF chat") # Solution : https://github.com/streamlit/streamlit/issues/1076 
            option = st.selectbox(label="Chat Option selection", options=options,
                            index=None,placeholder="Choose a chat type",
                            label_visibility="collapsed")
            if option == "New Chat": # index 0 
                self.app.create_session(chat_type=options.index(option))  # Create a new chat session
                st.session_state.active_page = "chat"
                st.session_state.layout = "centered"
                st.rerun()
            elif option == "Machine learning": # index 1
                self.app.create_session(chat_type=options.index(option))  # Create a machine learning session
                st.session_state.layout = "wide"
                st.rerun()
            elif option == "PDF chat": # index 2
                self.app.create_session(chat_type=options.index(option))  # Create a PDF chat session
                st.session_state.active_page = "chat"
                st.session_state.layout = "wide"
                st.rerun()

        with cols[1]:
            if st.button("📚 History"):
                st.session_state.active_page = "history"
                st.rerun()

        with cols[2]:
            if st.button("⚙️ Settings",key="home_settings"):
                self.render_settings_popover()

        self.render_features()
        self.render_history_page()

    def render_history_page(self):
        st.subheader("History")
        history_expander = st.expander("Recent history", expanded=True)

        with history_expander:
            # Display chat history (example: retrieve from `st.session_state` or fetch from database)
            if "sessions" in st.session_state:
                # Iterate through each session and create a "card" for each session
                for idx, session in enumerate(st.session_state.sessions):
                    # Go to the chat page button
                    chat_button = st.empty()
                    if chat_button.button(f"Chat {idx + 1}",key=f"{session}",use_container_width=True):
                        st.session_state.active_page = "chat"
                        self.app.switch_session(session_id = session)
                                
    @st.dialog("Settings")
    def render_settings_popover(self):
        """
        Render the settings popover when the settings button is clicked.
        """
        st.markdown("<h2>Settings</h2>", unsafe_allow_html=True)
        st.write("Here you can configure the chatbot settings.")
        
        # Placeholder settings options
        model_choice = st.selectbox("Choose Model", ["Simple Chat", "Chat with pdf", "Machine learning"])
        context_length = st.slider("Context Length", min_value=10, max_value=100, value=50)

        if st.button("Submit"):
            # Save the settings (example: store them in `st.session_state` or pass to `self.app`)
            st.session_state.chat_settings = {
                "model": model_choice,
                "context_length": context_length,
            }
            st.success("Settings updated successfully!")
            st.rerun()


