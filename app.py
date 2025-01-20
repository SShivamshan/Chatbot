import sys
sys.path.append(".")
import uuid
import os
import base64
from models.Model import Chatbot
from models.Agent import Agent
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from pages.home import HomePage
from pages.chat import ChatPage
from pages.history import HistoryPage
from pages.account import AccountManager
import logging
from datetime import datetime,timedelta
from collections import deque 
from threading import Thread
from queue import Queue
from typing import Callable
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_javascript import st_javascript
import base64
from models.RAG import RAG
from src.utils import *

# Solutions : https://medium.com/@b.antoine.se/building-a-custom-chatbot-a-streamlit-guide-to-ai-conversations-4ef524f0ea3f 
class ChatbotApp:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b", context_length: int = 18000):
        """Initialize the application and its pages."""
        
        self._setup_logging() # Logging setup

        # Initialize Streamlit session state
        if "sessions" not in st.session_state:
            st.session_state.sessions = {}
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = None
        if "active_page" not in st.session_state:
            st.session_state.active_page = "account"  # Start with the account Page
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "chat_counter" not in st.session_state:
            st.session_state.chat_counter = 0 # Ensures that the chat counter doesn't get augmented
        if "chat_type" not in st.session_state:
            st.session_state.chat_type = None # By default the chat type is a simple chat bot  ("New Chat" : 0, "Machine learning" : 1, "PDF chat" : 2)   
        if "layout" not in st.session_state:
            st.session_state.layout = "centered"  # Set the layout to wide mode Solution : https://discuss.streamlit.io/t/how-can-i-set-a-different-layout-per-page/81679 
        if "llm_instances" not in st.session_state: # Ensures that only one instance of each llm model is allowed and will be used in the session based on the data from the st.session_state.sessions
            st.session_state.llm_instances = {}

        self.messages_loaded = False
        self.current_page = "account"
        self.home_page = HomePage(self)
        self.chat_page = ChatPage(self)
        self.history_page = HistoryPage(self)
        self._base_url = base_url
        self._model = model
        self._context_length = context_length
    
        st.set_page_config(page_title="H1", initial_sidebar_state="collapsed",layout=st.session_state.layout)
        self.__vectorstore = Vectordb()
        self.account_page = AccountManager()
        if st.session_state.logged_in and not self.messages_loaded:
            st.session_state.user_id = self.account_page.get_user_id()
            self.on_login(st.session_state.user_id)
        self.__last_save_time = datetime.now()
        self.__SAVE_INTERVAL = timedelta(seconds=300)  # 5 minutes
        self.__MESSAGES_BEFORE_SAVE = 6
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def on_login(self, user_id: int):
        """Handle login and retrieve user-specific chats."""
        chat_manager = self.account_page.chats  # Get the ChatDataManager instance
        
        if self.messages_loaded:
            return  # Avoid reinitializing
        try:
            with chat_manager.session_scope() as session:
                chats = chat_manager.get_user_chats(user_id)
                for chat,chat_type,pdf_ref in chats:
                    if chat not in st.session_state.sessions:  # Only add new chats
                        messages = chat_manager.get_chat_history(chat)
                        st.session_state.chat_counter += 1
                        st.session_state.sessions[chat] = {
                            "name": f"Chat {len(st.session_state.sessions) + 1}",  # Assign default name
                            "messages":
                                deque( 
                                [
                                    {"User": msg["content"]} if msg["role"] == "User" else {"AI": msg["content"]}
                                    for msg in messages
                                ]
                            ), 
                            "chat_type" : chat_type,
                            "file_hash" : pdf_ref if chat_type == 2 else None,
                        }
                    if pdf_ref:
                        st.session_state.sessions[chat]["saved"] = True
                    if st.session_state.current_session_id is None:
                        st.session_state.current_session_id = chat 
               
            self.messages_loaded = True
        except Exception as e:
            st.error(f"Error retrieving user chats: {str(e)}")

    def create_session(self,chat_type:int=0):
        """Creates a new session with a default name."""
        if st.session_state.logged_in: # Verify if we are logged in 
            # if not st.session_state.current_session_id:  # Only create if no active session
            current_session_id = st.session_state.current_session_id
            if current_session_id:
                self.save_through_thread(func = self.handle_message_save, session_id = st.session_state.current_session_id,force=True)
            new_session_id = str(uuid.uuid4())
            st.session_state.chat_counter += 1
            new_session_name = f"Chat {st.session_state.chat_counter}"
            st.session_state.sessions[new_session_id] = {
                "name": new_session_name,
                "messages": deque(),
                "last_saved_index": -1,
                "chat_type": chat_type,
                "file_hash" : None # This will keep the 
            }
            st.session_state.current_session_id = new_session_id
            st.session_state.active_page = "chat"  # Set active page to "chat"
            st.session_state.chat_type = chat_type # Type of chat in question 

            if chat_type not in st.session_state.llm_instances:
                self.chatbot = self.create_llm(chat_type=chat_type)
            else:
                self.chatbot = st.session_state.llm_instances[chat_type] # Use the existing LLM instance

            st.session_state.layout = "wide" if chat_type == 2 else "centered"
            self.account_page.add_chat_id(new_session_id,chat_type=chat_type) # Add the session id to the user's account
            st.success(f"New session created: {new_session_name}")
            st.rerun()
        else:
            st.error("Cannot create a new session while another is active.")

    def switch_session(self, session_id: str):
        """Switches to an existing session without modifying the session name."""
        if session_id in st.session_state.sessions:
            # Save the current session before switching to the new session
            if st.session_state.current_session_id:
                self.save_through_thread(func = self.handle_message_save, session_id = st.session_state.current_session_id,force=True)
            # Switch to the new session
            st.session_state.current_session_id = session_id
            # Ensure the new session has a properly initialized last_saved_index
            if "last_saved_index" not in st.session_state.sessions[session_id]:
                st.session_state.sessions[session_id]["last_saved_index"] = -1  # Initialize if missing
            
            # Update active page and notify user
            st.session_state.active_page = "chat"
            st.session_state.layout = "wide" if st.session_state.sessions[st.session_state.current_session_id]["chat_type"] == 2 else "centered"
            chat_type = st.session_state.sessions[st.session_state.current_session_id]["chat_type"]
            if chat_type not in st.session_state.llm_instances:
                self.chatbot = self.create_llm(chat_type=chat_type)
            else:
                self.chatbot = st.session_state.llm_instances[chat_type]
                
            st.success(f"Switched to session: {st.session_state.sessions[session_id]['name']}")
            st.rerun()  # Trigger a rerun to refresh the page
        else:
            st.error("Session not found!")


    def delete_session(self, session_id: str):
        """Delete a session and update chat naming."""
        if session_id in st.session_state.sessions:
            # Frist delete the session from the db then remove the session itself
            deleted = self.account_page.chats.delete_chat(session_id)  # Remove the chat from the DB
            if not deleted:
                st.error(f"Failed to delete session {session_id} from the database.")
                return
            # Verify if that the current session id pointed towards the removed chat
            if session_id == st.session_state.current_session_id:
                st.session_state.current_session_id = None
            
            del st.session_state.sessions[session_id]
            # Recalculate chat numbers and update session names
            st.session_state.chat_counter = len(st.session_state.sessions)
            # print(st.session_state.chat_counter)
            for i, (_, session) in enumerate(st.session_state.sessions.items(), 1):
                session["name"] = f"Chat {i}"

            # Set to home page if no sessions left
            if not st.session_state.sessions:
                st.session_state.active_page = "home"
                st.session_state.current_session_id = None

            st.success(f"Session deleted: {session_id}")
            st.rerun()
        else:
            st.error("Session ID not found!")

    def get_image_base64(self,image_path): # Solution: https://discuss.streamlit.io/t/adding-a-link-to-my-image/53669/2
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    def customize_sidebar(self):
        st.logo("images/ai.png", size='large')
        st.sidebar.markdown("---")
        
        options=("New Chat", "Machine learning", "PDF chat") # Solution : https://github.com/streamlit/streamlit/issues/1076 
        option = st.sidebar.selectbox(label="Chat Option selection", options=options,
                        index=None,placeholder="Choose a chat type",
                        label_visibility="visible",
                        key="sidebar_chat_choice")
        if option == "New Chat": # index 0 
            self.app.create_session(chat_type=options.index(option))  # Create a new chat session
            st.session_state.active_page = "chat"
            st.session_state.layout = "centered"
            st.rerun()
        elif option == "Machine learning": # index 1
            self.app.create_session(chat_type=options.index(option))  # Create a machine learning session
            st.session_state.active_page = "chat"
            st.session_state.layout = "wide"
            st.rerun()
        elif option == "PDF chat": # index 2
            self.app.create_session(chat_type=options.index(option))  # Create a PDF chat session
            st.session_state.active_page = "chat"
            st.session_state.layout = "wide"
            st.rerun()

        if "sessions" in st.session_state:
            st.sidebar.subheader("Recent")
            # Sort sessions by name to maintain consistent order
            sorted_sessions = dict(sorted(st.session_state.sessions.items(), 
                                        key=lambda x: int(x[1]["name"].split()[-1])))
            for session_id, session_data in sorted_sessions.items():
                session_name = session_data["name"]
                is_active = session_id == st.session_state.current_session_id
                button_style = "primary" if is_active else "secondary"

                if st.sidebar.button(f"💬 {session_name}", 
                                key=f"session_{session_id}", 
                                type=button_style,
                                use_container_width=True):
                    self.switch_session(session_id)
                    st.rerun()

        st.sidebar.markdown("---")
        if st.sidebar.button("⚙️ Settings", key="settings_sidebar"):
            self.render_settings_popover()

        st.sidebar.markdown("---")
        st.sidebar.markdown("""
            <style>
            .logout-button {    
                position: fixed;
                bottom: 10px;
                width: 100%;
            }
            </style>
        """, unsafe_allow_html=True)

        if st.sidebar.button("🔓 Logout", key="logout_sidebar", help="Log out"):
            st.session_state.layout = "centered"
            if st.session_state.current_session_id:
                self.save_through_thread(func = self.handle_message_save, session_id = st.session_state.current_session_id,force = True)

            self.account_page.logout_db()

    @st.dialog("Settings")
    def render_settings_popover(self):
        """
        Render the settings popover when the settings button is clicked.
        """
        if st.session_state.logged_in:
            st.markdown("<h2>Settings</h2>", unsafe_allow_html=True)
            st.write("Here you can configure the chatbot settings.")
            
            # Placeholder settings options
            model_choice = st.selectbox("Choose Model", ["Simple Chat", "Chat with pdf", "Machine learning"])
            context_length = st.slider("Context Length", min_value=10, max_value=100, value=50)

            if st.button("Submit",key='Submit'):
                st.session_state.chat_settings = {
                    "model": model_choice,
                    "context_length": context_length,
                }
                st.success("Settings updated successfully!")
                st.rerun()

    def display_chat(self):

        chat_type = st.session_state.sessions[st.session_state.current_session_id]["chat_type"]
        if chat_type not in st.session_state.llm_instances:
            self.chatbot = self.create_llm(chat_type=chat_type)
        else:
            self.chatbot = st.session_state.llm_instances[chat_type]

        """Displays the chat for the current session."""
        if st.session_state.current_session_id and chat_type == 0:
            session = st.session_state.sessions[st.session_state.current_session_id]
            for message in session["messages"]:
                if "User" in message:
                    with st.chat_message("user"):
                        st.markdown(message["User"])
                if "AI" in message:
                    with st.chat_message("ai"):
                        st.markdown(message["AI"])

            # Input handling
            self.handle_input(chat_type=st.session_state.sessions[st.session_state.current_session_id]["chat_type"])

        elif st.session_state.current_session_id and chat_type == 2:
            col1, col2 = st.columns([5, 5], gap="large")  
            with col1:
                uploaded_file = st.file_uploader("Choose a PDF file")# accept_multiple_files=True

                if uploaded_file:
                    uploaded_file_hash = get_file_hash(uploaded_file)
                    
                    session = st.session_state.sessions[st.session_state.current_session_id]
                    # Check if the file is new or if embedding needs to be redone
                    if "file_hash" not in session or session["file_hash"] != uploaded_file_hash:
                        session["file_hash"] = uploaded_file_hash
                        session.pop("saved", None)  # Reset saved flag for new file

                    ui_width = st_javascript("window.innerWidth")
                    self.display_pdf(uploaded_file, width=ui_width - 10)

                    if not session.get("saved"):
                        with st.spinner("Processing embeddings..."):
                            saved = self.account_page.chats.add_pdf_ref(chat_id=st.session_state.current_session_id,pdf_ref=uploaded_file_hash)
                            # Solution : https://discuss.streamlit.io/t/st-toast-appears-now-on-the-top-right-corner/68854
                            # The toast appears for 4 seconds and on the top right corner
                            if saved:
                                st.toast("PDF reference added successfully")
                            else:
                                st.toast("Failed to add PDF reference.")
                            text = parse_pdf(uploaded_file) 
                            documents = split_chuncks(text=text)
                            done = self.__vectorstore.populate_vector(documents)
                            if done:
                                st.toast("Embedding and storing done successfully")
                                session["saved"] = True
                            else:
                                st.toast("Failed to embed and store the documents.")

            with col2:  # Chat section
                session = st.session_state.sessions[st.session_state.current_session_id]
                container_key = f"chat_container_{st.session_state.current_session_id}"
                message_container = st.container(height=750 ,key=container_key)
                with message_container: 
                    st.markdown("### Chat")
                    # Display existing messages
                    if "messages" in session:
                        # message_reverse = list(reversed(session["messages"]))
                        for message in session["messages"]:
                            if "User" in message:
                                with message_container.chat_message("user"):
                                    st.markdown(message["User"])
                            if "AI" in message:
                                with message_container.chat_message("ai"):
                                    st.markdown(message["AI"])
                # Handle new user input  
                self.handle_input(chat_type=session["chat_type"], container=message_container)
        else:
            st.write("No active session. Please create or switch to a session.")


    def create_llm(self, chat_type: int = 0):
        """Creates or retrieves the LLM for the specified chat type."""
        if chat_type in st.session_state.llm_instances:
            # Return the existing instance
            return st.session_state.llm_instances[chat_type]

        # Create a new instance
        chatbot = Chatbot(base_url=self._base_url, model=self._model, context_length=self._context_length)
        if chat_type == 2:  # Chat with PDF
            collection_size = self.__vectorstore.collection.count()
            retriever = self.__vectorstore.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": collection_size}
            )
            llm = RAG(vector_store=retriever, chatbot=chatbot)
        elif chat_type == 0:  # Simple chat
            memory = ConversationBufferMemory(return_messages=True)
            llm = ConversationChain(llm=chatbot, memory=memory)
        elif chat_type == 1: # Machine learning 
            # Handle other chat types here
            llm = None

        # Store the instance in session state
        st.session_state.llm_instances[chat_type] = llm
        return llm
 

    def handle_input(self,chat_type : int = 0, container = None):
        """Handles user input."""
        if chat_type == 0:  # Simple chat
            user_input = st.chat_input(placeholder="Ask me anything...",key="0")
            
            if user_input and st.session_state.current_session_id:
                session = st.session_state.sessions[st.session_state.current_session_id]
                if "last_saved_index" not in st.session_state.sessions[st.session_state.current_session_id]:
                    st.session_state.sessions[st.session_state.current_session_id]["last_saved_index"] = -1
                session["messages"].append({"User": user_input})
            
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                try:
                    # Get AI response
                    response = self.chatbot.invoke(input=user_input)["response"]
                    session["messages"].append({"AI": response})
                    # Display AI message
                    with st.chat_message("ai"):
                        st.markdown(response)

                    self.save_through_thread(func = self.handle_message_save, session_id = st.session_state.current_session_id)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        elif chat_type == 2:
            
            user_input = st.chat_input(placeholder="Ask me anything...",key="2")
            if user_input and st.session_state.current_session_id:
                session = st.session_state.sessions[st.session_state.current_session_id]
                if "last_saved_index" not in st.session_state.sessions[st.session_state.current_session_id]:
                    st.session_state.sessions[st.session_state.current_session_id]["last_saved_index"] = -1
                session["messages"].append({"User": user_input})
            
                # Display user message
                with container.chat_message("user"):
                    st.markdown(user_input)
                try:
                    with st.spinner("Thinking........"):
                        # Get AI response
                        response = self.chatbot.run(query=user_input)
                        session["messages"].append({"AI": response})
                        # Display AI message
                        with container.chat_message("ai"):
                            st.markdown(response)
                        self.save_through_thread(func = self.handle_message_save, session_id = st.session_state.current_session_id)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
    def display_pdf(self,file,width:int):  # Solution : https://discuss.streamlit.io/t/display-pdf-in-streamlit/62274 
        if st.session_state.sessions[st.session_state.current_session_id]:
            bytes_data = file.getvalue()
            base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
            # Embedding PDF in HTML
            pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{width*(3/4)}" type="application/pdf"></iframe>'
            # Displaying File
            st.markdown(pdf_display, unsafe_allow_html=True)

    def handle_message_save(self,session_id: str, force: bool = False):
        """Save chat state based on conditions or force save. The conditions are the following:
            - a time interval of 5 minutes
            - the number of messages in the chat exceeds a certain threshold (set to 5)
            - the user logs out 

        params
        ------
        session_id : str
            The current chat session id of the user. 
        force : bool, optional
            If True, force save the chat state regardless of the conditions.
        """
        current_time = datetime.now()
        session = st.session_state.sessions[session_id]
        if "last_saved_index" not in session:
            session["last_saved_index"] = -1

        save = force or \
                      len(session["messages"]) >= self.__MESSAGES_BEFORE_SAVE or \
                     (current_time - self.__last_save_time) >= self.__SAVE_INTERVAL
        last_saved_index = session["last_saved_index"]
        if save and st.session_state.logged_in:
            try:
                session = st.session_state.sessions[session_id]
                messages = list(session["messages"])
    
                # Get new messages (all messages after the last saved index)
                new_messages = messages[last_saved_index + 1 :]
                messages_to_save = [
                    {
                        "role": "User" if msg.get("User") else "AI",
                        "content": msg.get("User") or msg.get("AI")
                    }
                    for msg in new_messages
                ]
                success = self.account_page.chats.add_message(session_id, messages_to_save)
                
                if success:
                    self.__last_save_time = current_time  # Reset the timer
                    session["last_saved_index"] += len(new_messages)  # Update session-specific pointer
                    # print("After adding messages: ", session["last_saved_index"])
                else:
                    st.error("Failed to save chat messages.")
                    self.logger.error(f"Failed to save chat: {str(e)}")

            except Exception as e:
                st.error(f"Error saving chat: {str(e)}")
                self.logger.error(f"Error saving chat: {str(e)}")

    def save_through_thread(self, func:Callable, *args, **kwargs):   # Solution : https://github.com/streamlit/streamlit/issues/1326 , https://github.com/streamlit/streamlit/issues/8490 
        """
        Start a background thread to save messages or other elements
        """
        result_queue = Queue()

        def thread_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            result_queue.put(result)

        save_thread = Thread(target=thread_wrapper, args=args, kwargs=kwargs, daemon=True)
        add_script_run_ctx(save_thread)
        save_thread.start()

        # Wait for the thread to complete and retrieve the result
        save_thread.join()
        return result_queue.get() if not result_queue.empty() else None
    
    def run(self):
        """Run the application."""
        active_page = st.session_state.get("active_page", "account")  # Default to "account" page
        
        if not st.session_state.logged_in:
            self.account_page.render_account_page()  
        else:
            self.customize_sidebar()
            if active_page == "home":
                # print("Home page layout : {}".format(st.session_state.layout))
                self.home_page.render_home_page()
                
            elif active_page == "chat":
                if st.session_state.current_session_id:
                    # print("Home page layout : {}".format(st.session_state.layout))
                    if st.session_state.sessions[st.session_state.current_session_id]["chat_type"] == 0:
                        self.logger.info("Simple chat bot is created")
                        self.chat_page.render_chat_page()
                    elif st.session_state.sessions[st.session_state.current_session_id]["chat_type"] == 2:
                        self.logger.info("Pdf chat bot is created")
                        self.chat_page.render_pdf_chat_page()
                    elif st.session_state.sessions[st.session_state.current_session_id]["chat_type"] == 1:
                        self.logger.info("Machine learning chat bot is created")
                        self.chat_page.render_machine_learning_page()
                else:
                    st.error("No active chat session found. Please create a new chat from the sidebar.")
            elif active_page == "history":
                # st.session_state.layout = "centered"
                self.history_page.render_history_page()
            else:
                st.error("Unknown page!")