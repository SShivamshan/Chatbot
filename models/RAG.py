from typing import List, Optional, Dict,Any
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from models.Model import Chatbot
from pages.data_db import ChatDataManager
from src.utils import load_ai_template
from base64 import b64decode

class RAG(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llava:7b")
    context_length: int = Field(default=18000)
    multi_retriever: Optional[Any] = Field(default=None, exclude=True)
    memory: Optional[ConversationBufferMemory] = Field(default=None, exclude=True)  
    llm: Optional[Any] = Field(default=None, exclude=True)
    chat_data_manager: Optional[ChatDataManager] = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llava:7b",
        context_length: int = 18000,
        multi_retriever: Optional[Any] = None,
        chatbot: Optional[Any] = None,
        chat_data_manager: Optional[ChatDataManager] = None,
    ):
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            context_length=context_length
        )

        # Initialize LLM
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # Key used to track conversation history
            input_key="question",      # Key for the input user query
            output_key="response"      # Key for the output LLM response
        )
        self.multi_retriever = multi_retriever  # The retriever from Chroma for multi-document search
        self.chat_data_manager = chat_data_manager
    
    def load_chat_memory(self, chat_id: str):
        """Load prior chat history into memory."""
        if not self.chat_data_manager:
            raise ValueError("ChatDataManager is not initialized.")

        # Fetch chat history from the database
        history = self.chat_data_manager.get_chat_history(chat_id)

        # Inject into memory using a compact structure
        if history : 
            for message in history:
                self.memory.save_context(
                    {"question": message["content"] if message["role"] == "User" else None},
                    {"response": message["content"] if message["role"] == "AI" else None}
                )
    def summarize_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """ 
        Summarizes the chat history using a language model to condense past conversations. 
        """
        formatted_history = "\n".join(
            [f"User: {entry['question']}" if 'question' in entry else "" +
            f"AI: {entry['response']}" if 'response' in entry else "" 
            for entry in chat_history]
        )
        
        summarize_prompt = f"""Summarize the following conversation briefly. 
                                Respond only with the summary, no additional comment. 
                                Do not start your message by saying 'Here is a summary' or anything like that.
                                
                                {formatted_history}
                                """
        summarized_history = self.llm._call(summarize_prompt)
        return summarized_history
    
    def build_prompt(self,kwargs, config_path="config/config.yaml"):
        """
        Build a dynamic prompt using context, user question and prior chat history, with a template loaded from config.yaml.
        
        Parameters:
            kwargs (dict): Contains "context" (text/images), "question" and chat_history.
            config_path (str): Path to the YAML file with templates.

        Returns:
            ChatPromptTemplate: A formatted prompt ready for processing.
        """
        # Load template from config.yaml
        templates = load_ai_template(config_path)
        query_template = templates["Prompt_templates"]["RAG_templates"]["template"]

        docs_by_type = kwargs.get("context", {})
        user_question = kwargs.get("question", "")
        chat_history = kwargs.get("chat_history", "")
        working_history = chat_history

        if len(chat_history) > 6: # Summarize chat history
            working_history = self.summarize_chat_history(chat_history=chat_history)

        context_text = ""
        if "texts" in docs_by_type and docs_by_type["texts"]:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.page_content + "\n"

        template_variables = {
            "context": context_text,
            "question": user_question,
            "chat_history": working_history,  
        }
        prompt_template = query_template.format(**template_variables)

        # Construct the final prompt content
        prompt_content = [{"type": "text", "text": prompt_template}]

        # Include images if available
        if "images" in docs_by_type and docs_by_type["images"]:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                )

        # Return the formatted ChatPromptTemplate
        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )

    @staticmethod
    def parse_docs(docs):
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception as e:
                text.append(doc)
        return {"images": b64, "texts": text}

    def initialize_rag(self):
        
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.multi_retriever, # we use the mutlimodal retriever as the retiever to retieve all related form of results
            llm=self.llm,
        )
        # return multi_query_retriever
        try:
            retrieval_pipeline = {
                "context": multi_query_retriever | RunnableLambda(self.parse_docs),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: self.memory.load_memory_variables({}).get("chat_history", "")), 
            } | RunnablePassthrough().assign(
                response=(
                    RunnableLambda(self.build_prompt) 
                    | self.llm
                    | StrOutputParser()
                )
            )
            return retrieval_pipeline
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            return None  # Return None if error occurred during pipeline execution.  

    async def arun(self, query: str, chat_id: str = None) -> Dict:
        """Execute the RAG pipeline asynchronously."""
        # Load prior chat memory
        if chat_id:
            self.load_chat_memory(chat_id)

        rag_chain = self.initialize_rag()
        
        input_data = {
            "question": query,
            "chat_history": self.memory.load_memory_variables({}).get("chat_history", ""), 
        }
        
        response = await rag_chain.ainvoke(input_data)
        response_text = response if isinstance(response, str) else response.get("response", str(response))

        self.memory.save_context({"question": query}, {"response": response_text}) 
        return response

    def run(self, query:str, chat_id: str = None) -> Dict:
        """Execute the RAG pipeline synchronously."""
        # Load prior chat memory
        if chat_id:
            self.load_chat_memory(chat_id)

        rag_chain = self.initialize_rag()
        input_data = {
            "question": query,
            "chat_history": self.memory.load_memory_variables({}).get("chat_history", ""), 
        }
        response = rag_chain.invoke(input_data)
        response_text = response if isinstance(response, str) else response.get("response", str(response))

        self.memory.save_context({"question": query}, {"response": response_text}) # Solution : https://cheatsheet.md/langchain-tutorials/langchain-memory 
        return response 

        
