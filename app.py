import os
import sys
import uuid
import time
import base64
import logging
from datetime import datetime, timedelta
from collections import deque
from threading import Thread
from queue import Queue, Empty
from typing import Callable,List, Dict,Literal

import torch 
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_javascript import st_javascript
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain.chains import ConversationChain
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains.conversation.memory import ConversationBufferMemory

sys.path.append(".")
from models.Model import Chatbot
from models.Agent import Agent
from models.RAG import RAG
from pages.home import HomePage
from pages.chat import ChatPage
from pages.history import HistoryPage
from pages.account import AccountManager
from pages.data_db import ImageManager,TableManager
from src.utils import *


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
            st.session_state.chat_type = None # ("New Chat" : 0, "Agent" : 1, "PDF chat" : 2)   
        if "layout" not in st.session_state:
            st.session_state.layout = "centered"  # Set the layout to wide mode Solution : https://discuss.streamlit.io/t/how-can-i-set-a-different-layout-per-page/81679 
        if "llm_instances" not in st.session_state: # Ensures that only one instance of each llm model except for the the model for RAG which is created per session. 
            st.session_state.llm_instances = {}
        if "parent_directory" not in st.session_state:
            st.session_state.parent_directory = "images/img_output"

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
        self.pdf_reader = PDF_Reader()
        self.account_page = AccountManager()
        self.img_datadb = ImageManager(engine=self.account_page.engine)
        self.table_datadb = TableManager(engine=self.account_page.engine)
        if "image_data" not in st.session_state: 
            st.session_state.image_data = self.img_datadb.get_chat_image()  # Keeps in memory the as key-value pairs the extracted image path save and the image id as it's key
        if "table_data" not in st.session_state:
            st.session_state.table_data = self.table_datadb.get_chat_table() # Keep in memory the as key-value pairs the table id as key and the table html as it's value 
        if st.session_state.logged_in and not self.messages_loaded:
            st.session_state.user_id = self.account_page.get_user_id()
            self.on_login(st.session_state.user_id)
        self.__last_save_time = datetime.now()
        self.__SAVE_INTERVAL = timedelta(seconds=300)  # 5 minutes
        self.__MESSAGES_BEFORE_SAVE = 6

        self.AGENT_TAGS = {}
        
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
                for chat,chat_type,pdf_ref,pdf_filename in chats:
                    if chat not in st.session_state.sessions:  # Only add new chats
                        messages = chat_manager.get_chat_history(chat)
                        st.session_state.chat_counter += 1
                        message_deque = deque(
                            [
                                {"User": msg["content"]} if msg["role"] == "User" else {"AI": msg["content"]}
                                for msg in messages
                            ]
                        )
                        
                        # Initialize the session 
                        st.session_state.sessions[chat] = {
                            "name": f"Chat {len(st.session_state.sessions) + 1}",
                            "messages": message_deque,
                            "chat_type": chat_type,
                            "file_hash": pdf_ref if chat_type == 2 else None,
                            "filename" : pdf_filename if chat_type == 2 else None,
                            "last_saved_index": len(message_deque) - 1,  # Mark all loaded messages as saved
                            "from_history": set(range(len(message_deque))),  # Mark all message indices as from history
                        }
                    if pdf_ref:
                        st.session_state.sessions[chat]["saved"] = True
                    
               
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

            st.session_state.layout = "wide" if chat_type in [1,2] else "centered"
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
                # Offload the model from the current chat type when switching to sessions with different chat types
                chat_type = st.session_state.sessions[st.session_state.current_session_id]["chat_type"]
                if chat_type == 2 and st.session_state.llm_instances.get("chat_type", None):
                    if len(st.session_state.llm_instances[chat_type]) > 0:
                        st.session_state.llm_instances[chat_type][st.session_state.current_session_id].llm.unload_model()
                elif chat_type in [0,1] and len(st.session_state.llm_instances) != 0:
                    st.session_state.llm_instances[chat_type].llm.unload_model()
                self.save_through_thread(func = self.handle_message_save, session_id = st.session_state.current_session_id,force=True)
                
            # Switch to the new session
            st.session_state.current_session_id = session_id
            # Ensure the new session has a properly initialized last_saved_index
            if "last_saved_index" not in st.session_state.sessions[session_id]:
                st.session_state.sessions[session_id]["last_saved_index"] = -1  
            
            # Update active page and notify user
            st.session_state.active_page = "chat"
            st.session_state.layout = "wide" if st.session_state.sessions[st.session_state.current_session_id]["chat_type"] in [1,2] else "centered"
            # chat_type = st.session_state.sessions[st.session_state.current_session_id]["chat_type"]
                
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
        
        options=("New Chat", "Agent", "PDF chat") # Solution : https://github.com/streamlit/streamlit/issues/1076 
        option = st.sidebar.selectbox(label="Chat Option selection", options=options,
                        index=None,placeholder="Choose a chat type",
                        label_visibility="visible",
                        key="sidebar_chat_choice")
        if st.sidebar.button("Create Session"):
            if option == "New Chat": # index 0 
                self.create_session(chat_type=options.index(option))  # Create a new chat session
            elif option == "Agent": # index 1
                self.create_session(chat_type=options.index(option))  # Create a Agent session
            elif option == "PDF chat": # index 2
                self.create_session(chat_type=options.index(option))  # Create a PDF chat session

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
                # Offload all the models 
                self.unload_all_models()
                # self.save_through_thread(func = self.handle_message_save, session_id = st.session_state.current_session_id,force = True)

            self.account_page.logout_db()

    @st.dialog("Settings")
    def render_settings_popover(self):
        """
        Render the settings popover when the settings button is clicked.
        """
        general_tab, session_tab, other_tab = st.tabs(["Home", "Session", "Other"])
        with general_tab:
            st.header("General Settings")
            # Deleting chat history for this account
            # Log out of this account or delete this account
            # Change the localhost if there's a need to change 

        if st.session_state.active_page == "chat":
            with session_tab:
                st.markdown("<h2>Session settings</h2>", unsafe_allow_html=True)
                st.write("Here you can configure the chatbot settings.")
                params = {}
                model_name = None
                # Placeholder settings options
                chat_type = st.session_state.sessions[st.session_state.current_session_id]["chat_type"]
                model_choice = st.session_state.llm_instances.get(chat_type, None)
                if model_choice is None:
                    st.warning("Choose a model before setting it's parameters.")
                else:
                    if isinstance(model_choice, dict):
                        model_name = model_choice[st.session_state.current_session_id].llm.model
                    else:
                        model_name = model_choice.llm.model
                        
                    st.write(f"### Current Model Chosen: `{model_name}`")

                params = get_ollama_model_details(model_name=model_name) 
                if params:
                    # Read-only model parameters
                    st.markdown("### Model Details (Read-Only)")
                    col1, col2 = st.columns(2)  
                    with col1:
                        st.text_input("Architecture", params["architecture"], disabled=True)
                        st.text_input("Parameters", params["parameters"], disabled=True)
                    with col2:
                        st.text_input("Embedding Length", params["embedding_length"], disabled=True)
                        st.text_input("Quantization", params["quantization"], disabled=True)

                    # Editable parameter (context length)
                    st.markdown("### Modify Model Settings")
                    context_length = st.number_input(
                        "Context Length", min_value=100, max_value=params["context_length"], value=min(self._context_length, params["context_length"])
                    )

                temperature = st.slider("Temperature",min_value=0, max_value=10)

                if st.button("Submit",key='Submit'):
                    st.session_state.chat_settings = {
                        "context_length": context_length,
                        "temperature": temperature,
                        "model_name": model_name,
                    }
                    st.success("Settings updated successfully!")
                    st.rerun()
        with other_tab:
            st.header("Other")
            # API keys for OPENAI, adding new models
            
    @st.dialog("Embeddings saving...")
    def render_embeddings_popup(self, file:str, file_hash:str):
        id_key = "doc_id"
        with st.status(label="Processing the current document...",expanded=True,state="running") as status:
            session = st.session_state.sessions[st.session_state.current_session_id]
            filename = session.get("filename", None)
            if not self.account_page.chats.pdf_exist(filename):
                saved = self.account_page.chats.add_pdf_ref(chat_id=st.session_state.current_session_id,pdf_ref=file_hash,pdf_filename=filename)
                # Solution : https://discuss.streamlit.io/t/st-toast-appears-now-on-the-top-right-corner/68854
                # The toast appears for 4 seconds and on the top right corner so instead we do this 
                if saved:
                    st.write("PDF reference added successfully")
                else:
                    st.write("Failed to add PDF reference.")
                chunks = self.pdf_reader.read_pdf(file)
                texts, tables, images_64_list = self.pdf_reader.separate_elements(chunks)
                chunked_texts, doc_ids = split_chuncks(texts,filename)

                session["texts"] = chunked_texts
                session["doc_ids"] = doc_ids
                session["tables"] = tables
                session["images"] = images_64_list

                st.write("Document processed: Text, Tables, and Images extracted.")

                file_paths = self.save_through_thread(func=self._save_images, img=images_64_list,filename=filename)
                if file_paths is None:
                    self.logger.error("File paths were not retrieved from the thread")
                
                chatbot_image = Chatbot(model="llava:7b")
                chatbot_table = Chatbot()
                image_summaries = self.pdf_reader._get_summaries_image(images=images_64_list,chatbot=chatbot_image)
                table_summaries = self.pdf_reader._get_summaries_table(tables = tables,chatbot=chatbot_table)
                chatbot_image.unload_model()
                chatbot_table.unload_model()
                del chatbot_table
                del chatbot_image 

                if image_summaries or table_summaries:
                    st.write("Summaries have been generated successfully for images and tables")

                # Save the summaries and the text
                if chunked_texts:
                    self.__vectorstore.populate_vector(documents=chunked_texts)
                    st.write("Texts have been uploaded successfully")
                if image_summaries:
                    img_ids = [str(uuid.uuid4()) for _ in images_64_list]
                    session["img_ids"] = img_ids
                    summary_docs = [
                        Document(page_content=s, metadata={id_key: img_ids[i], "filename":filename})
                            for i, s in enumerate(image_summaries)
                    ]
                    self.__vectorstore.populate_vector(documents=summary_docs)
                    st.write("Image summaries have been uploaded successfully")
                    
                if table_summaries:
                    tab_ids = [str(uuid.uuid4()) for _ in tables]
                    session["tab_ids"] = tab_ids
                    table_docs = [
                        Document(page_content=s, metadata={id_key: tab_ids[i], "filename":filename})
                            for i, s in enumerate(table_summaries)
                    ]
                    self.__vectorstore.populate_vector(documents=table_docs)
                    st.write("Table summaries have been uploaded successfully")
                
                self._save_image_to_db() # Save the images to the db 
                self._save_table_to_db()
                unload_model(logger=self.logger, model_name="nomic-embed-text:latest") # For the embeddings used by the vectorstore
                # torch.cuda.empty_cache() # to remove the model used by the unstructured module 
                # Update Session State
                session["embeddings_ready"] = True
                session["saved"] = True

            time.sleep(1) # This is due to the fact that the update is done to fast
            status.update(  
                label="PDF processed!", state="complete"
            )

    def display_chat(self):

        session_id = st.session_state.current_session_id
        chat_type = st.session_state.sessions[session_id]["chat_type"]
        session = st.session_state.sessions[session_id]
        """Displays the chat for the current session."""
        if session_id and chat_type == 0:
            self.chatbot = self.get_or_create_llm(chat_type=chat_type)

            session = st.session_state.sessions[session_id]
            # self.display_messages(messages=session["messages"])
            for message in session["messages"]:
                if "User" in message:
                    with st.chat_message("user"):
                        st.markdown(message["User"])
                if "AI" in message:
                    with st.chat_message("ai"):
                        st.markdown(message["AI"])
            # Input handling
            self.handle_input(chat_type=chat_type)

        elif session_id and chat_type == 2:
            col1, col2 = st.columns([5, 5], gap="large")  
            with col1:
                uploaded_file = st.file_uploader("Choose a PDF file")# accept_multiple_files=True
                if uploaded_file:
                    uploaded_file_hash = get_file_hash(uploaded_file)
                    # self.img_datadb.delete_image()
                    # self.table_datadb.delete_table()
                    if "filename" not in session:
                        session["filename"] = get_pdf_title(uploaded_file.getvalue())
                    
                    # Check if the file is new or if embedding needs to be redone
                    if "file_hash" not in session or session["file_hash"] != uploaded_file_hash:
                        session["file_hash"] = uploaded_file_hash
                        if self.account_page.chats.pdf_exist(filename=session["filename"]): # The pdf already exists within our db 
                            session["embeddings_ready"] = False
                            session["saved"] = True
                            st.toast(" 🚨 PDF already present, using stored elements")
                        else: 
                            session["embeddings_ready"] = False
                            session.pop("saved", None)  # Reset saved flag for new file

                        # Invalidate the cached LLM instance if it exists
                        if chat_type == 2 and session_id in st.session_state.llm_instances.get(chat_type, {}):
                            st.session_state.llm_instances[2].pop(session_id)

                    if not session.get("saved"):
                        self.render_embeddings_popup(file=uploaded_file,file_hash=uploaded_file_hash)
                
                    ui_width = st_javascript("window.innerWidth")
                    self.display_pdf(uploaded_file, width=ui_width - 10)

                    self.chatbot = self.get_or_create_llm(chat_type, session_id)

            with col2:  # Chat section
                # session = st.session_state.sessions[st.session_state.current_session_id]
                container_key = f"chat_container_{session_id}"
                message_container = st.container(height=750 ,key=container_key)
                with message_container: 
                    st.markdown("### Chat")
                    # Display existing messages
                    self.display_messages(session["messages"], container=message_container)
                # Handle new user input  
                self.handle_input(chat_type=chat_type, container=message_container)

        elif session_id and  chat_type == 1:
            agent_col1, agent_col2 = st.columns([5, 5], gap="large") 
            with agent_col1:
                container_key = f"chat_container_agent_{session_id}"
                agent_message_container = st.container(height=800 ,key=container_key)
                with agent_message_container: 
                    st.markdown("### Chat")
                    # Display existing messages
                    self.display_messages(session["messages"], container=agent_message_container)
                self.handle_input(chat_type=chat_type, container=agent_message_container)
            with agent_col2:  # Reference and other information showing area updated each time 
                pass
        else:
            st.write("No active session. Please create or switch to a session.")

    def display_messages(self, messages, container=None):
        """Displays user and AI messages."""
        for message in messages:
            if container:
                with (container.chat_message("user") if "User" in message else container.chat_message("ai")):
                    st.markdown(message.get("User") or message.get("AI"))

    def get_or_create_llm(self, chat_type, session_id=None):
        """Retrieves or creates an LLM instance for the given chat type and session."""
        if chat_type not in st.session_state.llm_instances:
            st.session_state.llm_instances[chat_type] = {} if chat_type == 2 else None

        if chat_type == 2:
            # PDF-specific LLM tied to session_id
            if session_id not in st.session_state.llm_instances[chat_type]:
                st.session_state.llm_instances[chat_type][session_id] = self.create_llm(chat_type, session_id)
            return st.session_state.llm_instances[chat_type][session_id]
        else:
            # Shared LLM for chat_type 0 or 1
            if st.session_state.llm_instances[chat_type] is None:
                st.session_state.llm_instances[chat_type] = self.create_llm(chat_type)
            return st.session_state.llm_instances[chat_type]

    def create_llm(self, chat_type: int = 0, session_id: str = None):
        """Creates or retrieves the LLM for the specified chat type."""
        chatbot = Chatbot(base_url=self._base_url, model=self._model, context_length=self._context_length)
        if chat_type == 2:  # Chat with PDF
            # chatbot = Chatbot(model="llama3.2:3b") # Change to llama3.2
            llm = None
            # session_id = st.session_state.current_session_id
            session = st.session_state.sessions[st.session_state.current_session_id]
            embeddings_ready = session.get("embeddings_ready", False)
            saved = session.get("saved", False)

            if saved and embeddings_ready:
                docstore = st.session_state.get(f"docstore_{session_id}")
                if not docstore:
                    docstore = InMemoryStore()
                    st.session_state[f"docstore_{session_id}"] = docstore
                id_key = "doc_id"

                # Create the multi-vector retriever
                retriever = MultiVectorRetriever(
                    vectorstore=self.__vectorstore.vector_store,
                    docstore=docstore,
                    id_key=id_key,
                    search_type="similarity_score_threshold",
                    search_kwargs={"score_threshold": 0.19, "k" : 5},
                )
                if session.get("tables"):
                    retriever.docstore.mset(list(zip(session["tab_ids"], session["tables"])))
                if session.get("images"):
                    retriever.docstore.mset(list(zip(session["img_ids"], session["images"])))
                if session.get("texts"):
                    retriever.docstore.mset(list(zip(session["doc_ids"], session["texts"])))
                
                # print("docstore")
                llm = RAG(multi_retriever=retriever, chatbot=chatbot, chat_data_manager=self.account_page.chats)
            elif saved and not embeddings_ready:
                # print("no docstore")
                custom_retriever = CustomRetriever(vectorstore=self.__vectorstore.vector_store)
                llm = RAG(multi_retriever=custom_retriever, chatbot=chatbot , chat_data_manager=self.account_page.chats)

            return llm

        elif chat_type == 0:  # Simple chat
            # Create a new instance
            memory = ConversationBufferMemory(return_messages=True)
            llm = ConversationChain(llm=chatbot, memory=memory)
            return llm
        elif chat_type == 1: # AGENT
            # Handle other chat types here
            llm = None
            return llm
          
    def handle_input(self,chat_type : int = 0, container = None):
        """Handles user input."""
        if chat_type == 0:  # Simple chat
            user_input = st.chat_input(placeholder="Ask me anything...",key="0")
            
            if user_input and st.session_state.current_session_id:
                session = st.session_state.sessions[st.session_state.current_session_id]
                if "last_saved_index" not in session:
                    session["last_saved_index"] = -1
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
                if st.session_state.current_session_id in st.session_state.llm_instances.get(chat_type, {}):
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
                            response = self.chatbot.run(query=user_input , chat_id = st.session_state.current_session_id)
                            session["messages"].append({"AI": response["response"]})
                            self.render_multi_modal_response(response=response,container=container)
                            self.save_through_thread(func = self.handle_message_save, session_id = st.session_state.current_session_id)
                    except Exception as e:
                        self.logger.error("An error occured: {e}")
                        st.error(f"An error occurred: {e}")
                else:
                    message = "LLM instance is not created, to do so upload a PDF file"
                    self.popover_messages(message=message,msg_type="WARNING")
        else:
            st.markdown("Type `/link`, `/pdf` in the box below to see them highlighted.")
            user_input = st.chat_input(placeholder="Ask me anything...",key="1")

            if user_input and st.session_state.current_session_id:
                session = st.session_state.sessions[st.session_state.current_session_id]
                if "last_saved_index" not in st.session_state.sessions[st.session_state.current_session_id]:
                    st.session_state.sessions[st.session_state.current_session_id]["last_saved_index"] = -1

                flag, query = parse_flags_and_queries(input_text=user_input)
                # Verification phase for the flags and their sub flags if given 
                session["messages"].append({"User": user_input})

    @st.dialog(title="Session message")
    def popover_messages(self,message:str,msg_type:Literal["ERROR", "WARNING","INFO"] = str) : 
        if msg_type == "ERROR":
            st.error(message)
        elif msg_type == "WARNING":
            st.warning(message)
        elif msg_type == "INFO":
            st.info(message)
        else:
            st.info(message)


    def _save_images(self, img:List,filename:str):
        """Saves images from the current document"""
        session = st.session_state.sessions[st.session_state.current_session_id]
        try:
            base_path = os.getcwd()  # Get current working directory
            full_directory_path = os.path.join(base_path, st.session_state.parent_directory)

            # Ensure the directory exists, create it if not
            if not os.path.exists(full_directory_path):
                os.makedirs(full_directory_path)

            file_paths = []
            for i, image in enumerate(img):
                file_path = os.path.join(full_directory_path, f'{filename}_image_{i}.jpeg')

                # Check if the file already exists, skip if it does
                if os.path.exists(file_path):
                    self.logger.info(f'File already exists, skipping: {file_path}')
                    continue  

                # Convert base64 to image and process
                img_bytes = self.pdf_reader.base64_to_image(image)
                img = self.pdf_reader.create_image_from_bytes(img_bytes)
                # img = img.resize((100, 100), Image.LANCZOS)
                file_paths.append(file_path)
                img.save(file_path, optimize=True, quality=95)
                self.logger.info(f'Image saved to: {file_path}')
            # return file_paths
            session["saved_file_paths"] = file_paths
            return file_paths
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
        
    def _save_image_to_db(self):
        # Verify that the img_ids are not already in the db
        existing_image_data = st.session_state.get("image_data", {})
        existing_file_paths = set(existing_image_data.values()) 
        session = st.session_state.sessions[st.session_state.current_session_id]
        file_paths = session["saved_file_paths"]
        # Create a list of new file paths that are not already in the database
        new_file_paths = [file_path for file_path in file_paths if file_path not in list(existing_file_paths)]
        if not new_file_paths:
            return
        
        # Retrieve the current session
        new_images = {}
        img_ids_to_add = session.get("img_ids", [])  
        # Ensure that the number of new file paths matches the number of img_ids to add
        if len(new_file_paths) != len(img_ids_to_add):
            raise ValueError("The number of file paths does not match the number of img_ids.")
        new_images = dict(zip(img_ids_to_add, new_file_paths))
        if new_images:
            self.img_datadb.add_image(new_images) 

            # Update the session state with the new image db
            st.session_state.image_data = self.img_datadb.get_chat_image()

    def _save_table_to_db(self):
        # Add the tables to the db 
        session = st.session_state.sessions[st.session_state.current_session_id]
        tables_html = session["tables"]
        tab_ids = session["tab_ids"]

        # Create a dict of tab_ids : tab_html
        tab_dict = dict(zip(tab_ids, tables_html))

        # Add the tables to the db 
        if tab_dict:
            self.table_datadb.add_table(tables=tab_dict)

            # Update the session state with the new table db
            st.session_state.table_data = self.table_datadb.get_chat_table()

    def render_multi_modal_response(self, response:Dict, container= None):
        """Renders the chat message for images and text when dicussing with pdf"""
        # Get the text first 
        session = st.session_state.sessions[st.session_state.current_session_id]
        with container.chat_message("ai"):
            filename = session.get("filename", "unknown")
            if "response" in response:
                st.write(response["response"])
            
            if "context" in response and "texts" in response["context"]:
                st.markdown(
                    """
                    <style>
                    .custom-table table {
                        width: 100%!important;
                        border-collapse: collapse;
                    }
                    .custom-table th, 
                    .custom-table td {
                        border: 1px solid black;
                        padding: 5px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                data = []
                tab_data = []
                tab_ids = set(st.session_state.table_data.keys())
                for text in response["context"]["texts"]:
                    if isinstance(text, Document): # Documents thus texts and doc_ids pointing towards the table information 
                        metadata = text.metadata
                        if metadata.get("doc_id") not in tab_ids:
                            # Extract metadata details
                            page = metadata.get("page", "Unknown")
                            chunk = metadata.get("chunk", "Unknown")
                            source = metadata.get("source", "Unknown")
                            # Append metadata to the list
                            if all([page != "Unknown", chunk != "Unknown", source != "Unknown"]):
                                data.append({
                                    "Filename": filename,
                                    "Page": page,
                                    "Chunk": chunk,
                                    "Source (Excerpt)": source 
                                })
                        else:
                            doc_id = metadata.get("doc_id", None)
                            tab_data.append(st.session_state.table_data.get(doc_id)) # retrieve the table_html data from the db this means the pdf content is saved 
                        
                    elif isinstance(text, str): # STRING means we have the table in html format ex : <table><thead><tr><th> this is the case when the pdf is not saved
                        tab_data.append(text)
                        
                 # Convert to DataFrame and display it
                if data : 
                    st.write("**Relevant citations within the document:**")
                    df = pd.DataFrame(data)
                    st.write(df)
                if tab_data:
                    st.write("**Relevant tables present the document:**")
                    for table_html in tab_data:
                        st.markdown('<div class="custom-table">', unsafe_allow_html=True)
                        st.markdown(table_html, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

            # Finally the images
            st.markdown(
                """
                <style>
                .custom-image img {
                    width: 35% !important;
                    height: auto;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            if "context" in response and "images" in response["context"] and len(response["context"]["images"]) > 0:
                st.write("**Relevant images within the document:**")
                for image in response["context"]["images"]:
                    image_data = self.pdf_reader.base64_to_image(image)
                    st.markdown('<div class="custom-image">', unsafe_allow_html=True)
                    st.image(image_data, caption=f"Image from the article : {filename}", use_container_width=False)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Retrieve the doc_id from the response["context"]["texts"]
                doc_ids = set([doc.metadata.get("doc_id") for doc in response["context"]["texts"]])
                intersected_doc_ids = doc_ids.intersection(st.session_state.image_data.keys())
                if intersected_doc_ids:
                    st.write("**Relevant images within the document:**")
                    # Retrieve the corresponding image location from the db based on the doc_ids
                    for doc_id in intersected_doc_ids:
                        file_path = st.session_state.image_data[doc_id]
                        if file_path:
                            st.markdown('<div class="custom-image">', unsafe_allow_html=True)
                            st.image(file_path, width=450, caption=f"Image from the article : {filename}", use_container_width=False)
                            st.markdown("</div>", unsafe_allow_html=True)
            
    def display_pdf(self,file,width:int):  # Solution : https://discuss.streamlit.io/t/display-pdf-in-streamlit/62274 
        if st.session_state.sessions[st.session_state.current_session_id]:
            bytes_data = file.getvalue()
            base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

            pdf_width = int(width)
            pdf_height = int(pdf_width * (3/4))  # Maintain aspect ratio
            
            with st.container():
                # Add custom CSS to ensure proper containment
                st.markdown(
                    f"""
                    <style>
                        .pdf-container {{
                            max-width: {pdf_width}px;
                            overflow: hidden;
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display PDF with container class
                pdf_display = f"""
                    <div class="pdf-container">
                        <iframe 
                            src="data:application/pdf;base64,{base64_pdf}" 
                            width="{pdf_width}" 
                            height="{pdf_height}"
                            style="border: none;">
                        </iframe>
                    </div>
                """
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
        if "from_history" not in session:
            session["from_history"] = set()

        save = force or \
           len(session["messages"]) >= self.__MESSAGES_BEFORE_SAVE or \
           (current_time - self.__last_save_time) >= self.__SAVE_INTERVAL
    
        last_saved_index = session["last_saved_index"]
        
        if save and st.session_state.logged_in:
            try:
                messages = list(session["messages"])
                # Get only genuinely new messages (not from history)
                new_messages = [
                    msg for idx, msg in enumerate(messages[last_saved_index + 1:], start=last_saved_index + 1) # we take the messages + 1 from the last saved index
                    if idx not in session["from_history"] # we use the session["from_history"] to keep the number of messages that are already saved
                ]
                
                if new_messages:  # Only save if there are actually new messages
                    messages_to_save = [
                        {
                            "role": "User" if msg.get("User") else "AI",
                            "content": msg.get("User") or msg.get("AI")
                        }
                        for msg in new_messages
                    ]
                    
                    success = self.account_page.chats.add_message(session_id, messages_to_save)
                    
                    if success:
                        self.__last_save_time = current_time
                        session["last_saved_index"] = len(messages) - 1
                        self.logger.info(f"Saved {len(new_messages)} new messages for session {session_id}")
                    else:
                        st.error("Failed to save chat messages.")
                        self.logger.error(f"Failed to save chat messages for session {session_id}")
                else:
                    self.logger.debug(f"No new messages to save for session {session_id}")
                    
            except Exception as e:
                st.error(f"Error saving chat: {str(e)}")
                self.logger.error(f"Error saving chat for session {session_id}: {str(e)}")

    def save_through_thread(self, func:Callable, *args, **kwargs):   # Solution : https://github.com/streamlit/streamlit/issues/1326 , https://github.com/streamlit/streamlit/issues/8490 
        """
        Start a background thread to save messages or other elements
        """
        result_queue = Queue()

        def thread_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(None)
        save_thread = Thread(target=thread_wrapper, args=args, kwargs=kwargs, daemon=True)
        add_script_run_ctx(save_thread)
        save_thread.start()

        # Wait for the thread to complete and retrieve the result
        save_thread.join()
        try:
            return result_queue.get_nowait()  # Get result without blocking
        except Empty:
            self.logger.error("Queue is empty; no result returned from thread.")
            return None
    def unload_all_models(self):
        """Unload all loaded models when the app closes."""
        if "llm_instances" in st.session_state:
            if len(st.session_state["llm_instances"]) > 0:
                for chat_type, instances in st.session_state.llm_instances.items():
                    if chat_type == 2 and len(st.session_state["llm_instances"][chat_type]) > 0:  # PDF chat models (Multiple instances)
                        for _ , model in instances.items():
                            if model:
                                if model.llm.model == "llama3.2:3b":
                                    model.llm.unload_model()
                                    unload_model(logger=self.logger, model_name="nomic-embed-text:latest")
                                else:
                                    model.llm.unload_model()
                    elif instances:  # Chatbot models (Single instance)
                        instances.llm.unload_model()

        self.logger.info("All models have been unloaded from memory.")
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
                        self.logger.info("Agent is created")
                        self.chat_page.render_AGENT_page()
                else:
                    st.error("No active chat session found. Please create a new chat from the sidebar.")
            elif active_page == "history":
                # st.session_state.layout = "centered"
                self.history_page.render_history_page()
            else:
                st.error("Unknown page!")
        # Unload all the models if the user refreshes or closes the app 