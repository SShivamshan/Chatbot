import sys
import types
sys.path.append("..")
import os
from sqlalchemy import create_engine, text, Column, String, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
import logging
from contextlib import contextmanager
import streamlit as st
from pages.data_db import ChatDataManager
from pages.base import Base

class Account(Base):
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

class AccountManager:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._setup_logging()
            self.engine = create_engine('sqlite:///database/app.db')
            self.SessionFactory = sessionmaker(bind=self.engine)
            Base.metadata.create_all(self.engine)
            self.chats = ChatDataManager(engine=self.engine)
            self.initialized = True

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    @contextmanager
    def session_scope(self):
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def authenticate_user(self, username: str, password: str) -> bool:
        try:
            with self.session_scope() as session:
                user = session.query(Account).filter_by(
                    username=username, 
                    password=password
                ).first()
                if user: # Keeps in memory the current user and his id
                    self.username = username 
                    self.__id = user.id
                    return True
            return False
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return False

    def create_account(self, account: dict) -> None:
        try:
            with self.session_scope() as session:
                new_account = Account(
                    username=account['username'],
                    password=account['password']
                )
                session.add(new_account)
                session.commit()
                st.success(f"Account for {account['username']} created successfully!")
                st.balloons()
                st.cache_data.clear()
                st.rerun()
        except IntegrityError:
            st.warning(f"Username '{account['username']}' is already taken.")
        except Exception as e:
            st.error(f"Error creating account: {e}")
            # logging.exception("Account creation failed")
            self.logger.exception("Account creation failed")

    def render_account_page(self):
        st.title("Welcome to H1")
        st.subheader("Login or Create a New Account")
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        with login_tab:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", key='login'):
                if self.authenticate_user(username, password):
                    st.success(f"Welcome back, {username}!")
                    st.session_state.logged_in = True
                    st.session_state.active_page = "home"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with signup_tab:
            new_username = st.text_input("New Username", key="signup_username")
            new_password = st.text_input("New Password", type="password", key="signup_password")
            if st.button("Sign Up", key='signup'):
                if new_username and new_password:
                    self.create_account({
                        "username": new_username, 
                        "password": new_password
                    })
                else:
                    st.warning("Please provide both username and password.")

    def logout_db(self):
        # Clear session-specific states
        st.session_state.logged_in = False
        st.session_state.active_page = "account"
        st.session_state.current_session_id = None
        st.session_state.sessions = {}
        st.session_state.chat_counter = 0
        
        try:
            with self.session_scope() as session:
                self.engine.close()
                self.engine.dispose()
            st.success("Successfully logged out.")
        except Exception as e:
            st.error(f"Error during logout: {e}")

        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
        
    def delete_user(self, user_id):
        pass

    def add_chat_id(self,session_id: str, chat_type: int):
        """
        Adds the chat id from the created session to the chat data base
        """
        self.chats.add_session_chat_id(user_id=self.__id,session_id=session_id,chat_type=chat_type)

    def get_user_id(self):
        return self.__id
    
