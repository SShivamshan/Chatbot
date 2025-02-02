import sys
from typing import List,Dict 
sys.path.append("..")
import os
from sqlalchemy import Column, String, Integer,ForeignKey,DateTime,Text
from sqlalchemy.orm import relationship,sessionmaker
import logging
from contextlib import contextmanager
import streamlit as st
from datetime import datetime, timezone
from pages.base import Base
from collections import deque
## Solution adapted from: https://ploomber.io/blog/streamlit-postgres/ 

class ChatData(Base):
    __tablename__ = 'chat_data'
    chat_id = Column(String, primary_key=True)
    chat_type = Column(Integer,nullable=False)
    pdf_ref = Column(String,nullable=True)
    pdf_filename = Column(String,nullable=True)
    user_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc)) 
    messages = relationship('MessageData', back_populates='chat', cascade='all, delete-orphan') # 

class MessageData(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    chat_id = Column(String, ForeignKey('chat_data.chat_id', ondelete='CASCADE'))
    user_prompt = Column(Text,nullable=True)  
    ai_prompt = Column(Text,nullable=True)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    chat = relationship('ChatData', back_populates='messages')
    
class ImageData(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    img_id = Column(String, nullable=False)
    img_url = Column(String, nullable=False)

class TableData(Base):
    __tablename__ = 'tables'
    id = Column(Integer, primary_key=True)
    table_id = Column(String, nullable=False)
    table_html = Column(String, nullable=False)

class ChatDataManager:
    _instance  =  None
    
    def __new__(cls,*args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,engine = None) -> None:
        if not hasattr(self, 'initialized'):
            self._setup_logging()
            self.engine = engine
            self.SessionFactory = sessionmaker(bind=self.engine)
            ChatData.metadata.create_all(self.engine)
            MessageData.metadata.create_all(self.engine)
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
            self.logger.error(f"Database error: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()

    def add_session_chat_id(self, user_id: int, session_id: str,chat_type: int) -> bool:
        try:
            with self.session_scope() as session:
                chat_data = ChatData(chat_id=session_id, chat_type=chat_type ,user_id=user_id)
                session.add(chat_data)
            
            st.cache_data.clear()
            st.rerun()
            return True
        except Exception as e:
            self.logger.error(f"Failed to add chat session: {str(e)}", exc_info=True)
            st.error(f"Error adding chat session: {str(e)}")
            return False

    def get_user_chats(self, user_id: int) -> list[str]:
        with self.session_scope() as session:
            return [(chat.chat_id, chat.chat_type, chat.pdf_ref, chat.pdf_filename) for chat in session.query(ChatData).filter_by(user_id=user_id).all()]

    def delete_chat(self, chat_id: str) -> bool:
        try:
            with self.session_scope() as session:
                chat_to_delete = session.query(ChatData).filter_by(chat_id=chat_id).first()
                if not chat_to_delete:
                    self.logger.info(f"No chat found with ID {chat_id}.")
                    return False
                
                session.delete(chat_to_delete)
                # session.flush()
                # session.commit()
                self.logger.info(f"Chat with ID {chat_id} successfully deleted from the database.")
                return True
        except Exception as e:
            self.logger.error(f"Failed to delete chat: {str(e)}", exc_info=True)
            return False
        
    def add_pdf_ref(self,chat_id: str,pdf_ref :str,pdf_filename: str) -> bool:
        try:
            with self.session_scope() as session:
                chat_to_update = session.query(ChatData).filter_by(chat_id=chat_id).first()
                if not chat_to_update:
                    self.logger.info(f"No chat found with ID {chat_id}.")
                    return False
                chat_to_update.pdf_ref = pdf_ref
                chat_to_update.pdf_filename = pdf_filename
                # session.commit()
                self.logger.info(f"PDF reference added for chat with ID {chat_id}.")
                return True
        except Exception as e:
            self.logger.error(f"Failed to add PDF reference: {str(e)}", exc_info=True)
            return False
        
    def add_message(self, chat_id: str, messages: list[dict]) -> bool:
        """
        Add one or more messages to the database.

        params
        ------
        chat_id : str
            The ID of the chat to which the messages belong.
        messages : list[dict]
            A list of dictionaries, each representing a message with keys 'role' and 'content'.

        returns
        -------
        bool
            True if messages were successfully added, False otherwise.
        """
        try:
            with self.session_scope() as session:
                message_objects = []
                i = 0
                while i < len(messages):
                    # Extract the User message
                    if messages[i]["role"] == "User":
                        user_message = messages[i]["content"]
                    else:
                        user_message = None

                    # Extract the AI message
                    if i + 1 < len(messages) and messages[i + 1]["role"] == "AI":
                        ai_message = messages[i + 1]["content"]
                    else:
                        ai_message = None

                    if user_message or ai_message:
                        message_objects.append(
                            MessageData(chat_id=chat_id, user_prompt=user_message, ai_prompt=ai_message)
                        )

                    # Move to the next message (increment by 2 since we process in pairs)
                    i += 2
                
                # Add all MessageData objects to the session
                session.add_all(message_objects)
                # session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to add messages: {str(e)}", exc_info=True)
            return False

    def get_chat_history(self, chat_id: str) -> list[dict]:
        with self.session_scope() as session:
            messages = session.query(MessageData)\
                .filter_by(chat_id=chat_id)\
                .order_by(MessageData.timestamp)\
                .all()
            history = deque()
            for msg in messages:
                if msg.user_prompt is not None:  # User's message
                    history.append({"role": "User", "content": msg.user_prompt})
                if msg.ai_prompt is not None:  # AI's response
                    history.append({"role": "AI", "content": msg.ai_prompt})

            return history
        
    def pdf_exist(self,filename) -> bool:
        """
        Verify if the pdf exists within all the sessions 
        """
        try:
            with self.session_scope() as session:
                chats = session.query(ChatData).all()
                for chat in chats:
                    if chat.pdf_filename: 
                        if chat.pdf_filename.lower() == filename.lower():
                            return True
                    else:
                        return False
            return False
        except Exception as e:
            self.logger.error(f"Failed to verify pdf existence: {str(e)}", exc_info=True)
            return False
        
class ImageManager:
    _instance  =  None
    
    def __new__(cls,*args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,engine = None):
        if not hasattr(self, 'initialized'):
            self._setup_logging()
            self.engine = engine
            self.SessionFactory = sessionmaker(bind=self.engine)
            ImageData.metadata.create_all(self.engine)
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
            self.logger.error(f"Database error: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()

    def add_image(self, img_data:dict):
        """
        Add a new image reference to the database.
        """
        try:
            with self.session_scope() as session:
                img_objects = []
                for img_id,img_url in img_data.items():
                    img_objects.append(
                        ImageData(img_id = img_id, img_url = img_url)
                    )
                
                session.add_all(img_objects)
        except Exception as e:
            self.logger.error(f"Failed to add messages: {str(e)}", exc_info=True)
            raise Exception(f"Failed to add image: {str(e)}")
    
    def get_chat_image(self, img_ids:List = None) -> List[str]:
        """
        Get all images associated to the img_ids.
        """
        try:
            with self.session_scope() as session:
                if img_ids is None:
                    img_data = session.query(ImageData).all()
                else:
                    img_data = session.query(ImageData)\
                        .filter(ImageData.img_id.in_(img_ids))\
                        .all()
                
                return {img.img_id: img.img_url for img in img_data}
        except Exception as e:
            self.logger.error(f"Failed to get image data: {str(e)}", exc_info=True)
            return None
        

    def delete_image(self):
        """
        Delete all images from the database.
        """
        try:
            with self.session_scope() as session:
                session.query(ImageData).delete()
        except Exception as e:
            self.logger.error(f"Failed to delete images: {str(e)}", exc_info=True)
            raise Exception(f"Failed to delete images: {str(e)}")
        

class TableManager:
    _instance  =  None
    
    def __new__(cls,*args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,engine = None):
        if not hasattr(self, 'initialized'):
            self._setup_logging()
            self.engine = engine
            self.SessionFactory = sessionmaker(bind=self.engine)
            TableData.metadata.create_all(self.engine)
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
            self.logger.error(f"Database error: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()
            
    def add_table(self,tables:Dict):
        """
        Add a new table reference to the database.
        """
        try:
            with self.session_scope() as session:
                table_objects = []
                for table_id,table_html in tables.items():
                    table_objects.append(
                        TableData(table_id = table_id, table_html = table_html)
                    )
                
                session.add_all(table_objects)
        except Exception as e:
            self.logger.error(f"Failed to add table: {str(e)}", exc_info=True)
            raise Exception(f"Failed to add table: {str(e)}")

    def get_chat_table(self, table_ids:List = None) -> List[str]:
        """
        Get all tables associated to the table_ids.
        """
        try:
            with self.session_scope() as session:
                if table_ids is None:
                    table_data = session.query(TableData).all()
                else:
                    table_data = session.query(TableData)\
                        .filter(TableData.table_id.in_(table_ids))\
                        .all()
                
                return {table.table_id: table.table_html for table in table_data}
        except Exception as e:
            self.logger.error(f"Failed to get table data: {str(e)}", exc_info=True)
            return None
        
    
    def delete_table(self):
        """
        Delete all tables from the database.
        """
        try:
            with self.session_scope() as session:
                session.query(TableData).delete()
        except Exception as e:
            self.logger.error(f"Failed to delete tables: {str(e)}", exc_info=True)
            raise Exception(f"Failed to delete tables: {str(e)}")