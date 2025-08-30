import sys
sys.path.append(".")
import time
import subprocess
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
                "icon": "💬",
                "title": "Chat",
                "description": "Interactive chatbot for conversations, questions, and general assistance."
            },
            {
                "icon": "📄",
                "title": "RAG",
                "description": "Upload and chat with your PDF documents using retrieval-augmented generation."
            },
            {
                "icon": "🤖",
                "title": "Agent",
                "description": "Advanced agentic capabilities including RAG agents, code agents, and knowledge-based agents."
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
            options=("Chat", "Agent", "PDF chat") # Solution : https://github.com/streamlit/streamlit/issues/1076 
            option = st.selectbox(label="Chat Option selection", options=options,
                            index=None,placeholder="Choose a chat type",
                            label_visibility="collapsed")
            if option == "Chat": # index 0 
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
            # Display chat history
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
        general_tab, other_tab = st.tabs(["Home", "Other"]) # Here the session tab will allow the user to see the different type of llm already in use(llm_instances)
        with general_tab:
            st.header("General Settings")
            # Now we need to add what they do directly on the right 
            delete_tab, logout, clear_tab = st.columns(3)
            # Deleting chat history for this account
            if delete_tab.button("Delete Chat", key="delete_button", help="Permanently remove all chat history for this account"):
                user_chats =  self.account_page.chats.get_user_chats(st.session_state.user_id)

                if not user_chats:
                    st.info("No chats found for this account.")
                else:
                    success_count = 0
                    for chat in user_chats:
                        chat_id = chat[0]  # chat_id is the first element in the tuple
                        if self.account_page.chats.delete_chat(chat_id):
                            success_count += 1
                    
                    st.success(f"Deleted {success_count} chat(s) for this account.")
                    if "sessions" in st.session_state:
                        st.session_state.sessions.clear()
                        st.session_state.chat_counter = 0 

                    if not st.session_state.sessions:
                        st.session_state.active_page = "home"
                        st.session_state.current_session_id = None
                        st.rerun()
                
            st.divider()  
            if logout:
                col1, col2 = st.columns(2)

                if col1.button("🔓 Logout", key="logout_settings_popover", help="Logs out this account"):
                    st.session_state.layout = "centered"
                    if st.session_state.get("current_session_id"):
                        self.unload_all_models()
                    self.account_page.logout_db()

                if col2.button("Delete Account", key="delete_account_settings_popover", help="Permanently delete this account and all associated data"):
                    st.session_state.layout = "centered"
                    if st.session_state.get("current_session_id"):
                        self.unload_all_models()
                    self.account_page.delete_user(st.session_state.user_id)
                    self.account_page.logout_db()

            st.divider() 
            if clear_tab.button("Clear All Data", key="clear_button", help="Permanently remove all chats, images, and tables for this account"):
                user_id = st.session_state.user_id
                user_chats = self.account_page.chats.get_user_chats(user_id)
                chat_count = 0
                if user_chats:
                    for chat in user_chats:
                        chat_id = chat[0]
                        if self.account_page.chats.delete_chat(chat_id):
                            chat_count += 1
                try:
                    self.img_datadb.delete_image(user_id)
                    img_deleted = True
                except Exception as e:
                    img_deleted = False
                    st.error(f"Failed to delete images: {str(e)}")

                try:
                    self.table_datadb.delete_table(user_id)
                    tbl_deleted = True
                except Exception as e:
                    tbl_deleted = False
                    st.error(f"Failed to delete tables: {str(e)}")

                st.success(f"Deleted {chat_count} chat(s), "
                        f"{'all images' if img_deleted else 'no images'}, "
                        f"{'all tables' if tbl_deleted else 'no tables'} for this account.")
                time.sleep(1)

                if "sessions" in st.session_state:
                    st.session_state.sessions.clear()
                    st.session_state.chat_counter = 0 

                if not st.session_state.sessions:
                    st.session_state.active_page = "home"
                    st.session_state.current_session_id = None
                    st.rerun()

        with other_tab:
            st.header("Other / ADD")

            option = st.radio("Choose a model source:", ["OpenAI", "Ollama"])

            if option == "OpenAI":
                st.subheader("🔑 OpenAI API Key")
                api_key = st.text_input("Enter your OpenAI API Key:", type="password")
                if api_key:
                    st.session_state["openai_api_key"] = api_key
                    st.success("✅ OpenAI API Key saved! (session only)")
                    st.rerun()

            elif option == "Ollama":
                st.subheader("🦙 Ollama Models")

                model_name = st.text_input(
                    "Enter the name of the Ollama model you want to pull (e.g., llama2, mistral, codellama):"
                )

                if st.button("Pull Model"):
                    if model_name.strip():
                        try:
                            with st.spinner(f"Pulling model `{model_name}` from Ollama registry..."):
                                pull_result = subprocess.run(
                                    ["ollama", "pull", model_name],
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                            st.session_state["ollama_model"] = model_name
                            st.success(f"✅ Model `{model_name}` pulled successfully!")
                            st.rerun()
                        except subprocess.CalledProcessError as e:
                            st.error(f"❌ Failed to pull model: {e.stderr}")
                    else:
                        st.warning("⚠️ Please enter a model name before pulling.")

            


