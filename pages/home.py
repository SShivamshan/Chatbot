import sys
sys.path.append(".")
import streamlit as st
from datetime import datetime

class HomePage:
    def __init__(self, app):
        """
        Initialize with a reference to the main app.

        params
        ------ 
        - app: The main ChatbotApp instance.
        """
        self.app = app
        

    def render_welcome_section(self,username:str):
        """
        Render the welcome portion of the home page 
        
        params
        ------
        - username (str): 
            Username of the user who is logged in.
        
        """
        greeting = self.get_greeting()
        st.markdown(f"<h1>🤖 Welcome to H1, {username} </h1>", unsafe_allow_html=True)
        st.markdown(f"<p>{greeting}! I'm H1, your AI assistant. How can I help you today?</p>", unsafe_allow_html=True)

    def get_greeting(self) -> str:
        """
        Get time-based greeting.
        
        returns
        -------
            str: Greeting based on current time of day.
        """
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good Morning"
        elif 12 <= hour < 17:
            return "Good Afternoon"
        else:
            return "Good Evening"
        
    def render_features(self):
        """
        Render feature highlights in a horizontal layout using columns.
        """
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
            options=("New Chat", "Agent", "PDF chat") # Solution : https://github.com/streamlit/streamlit/issues/1076 
            option = st.selectbox(label="Chat Option selection", options=options,
                            index=None,placeholder="Choose a chat type",
                            label_visibility="collapsed")
            if option == "New Chat": # index 0 
                self.app.create_session(chat_type=options.index(option))  # Create a new chat session
                # st.rerun()
            elif option == "Agent": # index 1
                self.app.create_session(chat_type=options.index(option))  # Create a Agent session
                # st.rerun()
            elif option == "PDF chat": # index 2
                self.app.create_session(chat_type=options.index(option))  # Create a PDF chat session
                # st.rerun()

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
        """
        Render the history page present in the home page. 
        """
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
        general_tab, session_tab, other_tab = st.tabs(["Home", "Session", "Other"]) # Here the session tab will allow the user to see the different type of llm already in use(llm_instances)
        with general_tab:
            st.header("General Settings")
            # Now we need to add what they do directly on the right 
            delete_tab, logout, change = st.columns(3)
            # Deleting chat history for this account
            if delete_tab.button("Delete Chat", key="delete_button", help="Permanently remove all chat history for this account"):
                pass
            
            st.divider()  
            if logout:
                col1, col2 = st.columns(2)

                col1.button("🔓 Logout", key="logout_settings_popover", help="Logs out this account")
                col2.button("Delete Account", key="delete_account_settings_popover", help="Permanently delete this account and all associated data")
            
            st.divider() 
            # Change the localhost if there's a need to change 
            local_host = st.text_input("Localhost",value="8501",key="localhost")
            if st.button("Submit",key="submit_settings_popover"):
                st.session_state.local_host = local_host

        with session_tab:
            st.header("Session settings")
            st.write("Here you can observe the chatbot settings that are already in use since you logged.")
            params = {}

            


