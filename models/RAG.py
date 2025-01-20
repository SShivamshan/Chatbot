from typing import List, Optional, Dict,Any
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.schema import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import format_document,Document
from langchain_core.messages import get_buffer_string
from models.Model import Chatbot
from operator import itemgetter


class RAG(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    ai_template: dict = Field(default_factory=dict)
    vector_store: Optional[BaseRetriever] = Field(default=None, exclude=True)
    memory: Optional[ConversationBufferMemory] = Field(default=None, exclude=True)
    llm: Optional[Any] = Field(default=None, exclude=True)
    document_prompt: PromptTemplate = Field(
        default=PromptTemplate.from_template(template="{page_content}")
    )

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        context_length: int = 18000,
        vector_store: Optional[BaseRetriever] = None,
        chatbot: Optional[Any] = None
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

        self.vector_store = vector_store # The retriever from Chroma 
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"  # Set input_key to align with combined input
        )

    def get_query_prompt(self) -> PromptTemplate:
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question", "chat_history"],  # Now includes chat_history
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question, considering the conversation history.
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
            
            Provide these alternative questions separated by newlines.
            
            Original question: {question}
            Chat history: {{chat_history}}""", 
        )

        return QUERY_PROMPT
    
    def summarize_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Summarizes the chat history using a language model to condense past conversations.
        """
        # Format the chat history to give the model context on how to summarize
        formatted_history = "\n".join(
            [f"User: {msg['content']}" if msg['role'] == 'user' 
            else f"AI: {msg['content']}" for msg in chat_history]
        )

        # Summarization prompt
        summarize_prompt = f"Summarize the following conversation briefly:\n\n{formatted_history}"

        # Pass the prompt to the LLM (assuming self.llm is a callable LLM instance)
        summarized_history = self.llm(summarize_prompt)

        return summarized_history
    
    def _combine_documents(
        self,
        docs: List[Document],
        document_separator: str = "\n\n"
    ) -> str:
        """Combine multiple documents into a single string."""
        doc_strings = [format_document(doc,self.document_prompt) for doc in docs]# self.document_prompt.format(**doc.model_dump())
        return document_separator.join(doc_strings)
    
    def build_prompt(self, kwargs: dict) -> list:
        """
        Dynamically build the prompt by merging chat history with retrieved context.
        """
        # Extract context and chat history from kwargs
        context = kwargs.get("context", [])  # Expecting a list of Documents
        chat_history = kwargs.get("chat_history", [])

        # If chat history is too long, summarize it
        if len(chat_history) > 5:  # Example: Summarize after 5 messages
            summarized_chat_history = self.summarize_chat_history(chat_history)
        else:
            # Use raw chat history if it's small
            summarized_chat_history = "\n".join(
                [f"User: {msg['content']}" if msg['role'] == 'user' 
                else f"AI: {msg['content']}" for msg in chat_history]
            )

        # Combine documents into formatted context
        formatted_context = self._combine_documents(context) if context else ""

        # Merge summarized chat history with the formatted context
        full_context = (
            f"{summarized_chat_history}\n\n{formatted_context}" 
            if summarized_chat_history 
            else formatted_context
        )

        query_template = """Answer the question based ONLY on the following context and rules:

        Context: {context}
        
        Rules:
        1. Always base your answers on the content of the provided context.
        2. If asked for analysis or observations, structure your response with clear headings and bullet points for clarity.
        3. Use your own knowledge for the given context at the end to add information if this is useful.
        4. If the information isn't available in the provided context, clearly state this and avoid speculation.
        5. When appropriate, suggest further areas of investigation or additional data that might be useful.
        6. Provide a detailed and concise answer.
        7. Use the provided answer format to format the answer.

        Answer Format:
        Doc Information:
        [Information directly from the provided documents]

        Additional Relevant Information:
        [Any relevant additional context or knowledge]

        Question: {question}
        """

        prompt_template = ChatPromptTemplate.from_template(query_template)

        return prompt_template.format_messages(
            context=full_context,
            question=kwargs["question"]
        )

    def initialize_rag(self):
        QUERY_PROMPT = self.get_query_prompt()

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history")
        )

        # Initialize MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store,
            llm=self.llm,
            prompt=QUERY_PROMPT,
        )

        query_generation = { # Generate an advanced query based on the question and the chat history 
            "retrieved_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"])
            } | QUERY_PROMPT | self.llm
        }

        document_retrieval = {
            "context": itemgetter("retrieved_question") | retriever,
            "question": lambda x: x,
        }
        
        final_chain = (
            loaded_memory 
            | query_generation 
            | document_retrieval 
            | RunnableLambda(self.build_prompt)
            | self.llm 
            | StrOutputParser()
        )

        return final_chain

    async def arun(self, query: str) -> str:
        """Execute the RAG pipeline asynchronously."""
        rag_chain = self.initialize_rag()
        
        input_data = {
            "question": query,
            "chat_history": self.memory.chat_memory.messages
        }
        
        response = await rag_chain.ainvoke(input_data)
        
        if self.memory:
            await self.memory.save_context(
                inputs={"question": query},
                outputs={"response": response}
            )
            
        return response

    def run(self, query: str) -> str:
        """Execute the RAG pipeline synchronously."""
        rag_chain = self.initialize_rag()

        input_data = {
            "question": query,
            "chat_history": self.memory.chat_memory.messages
        }

        response = rag_chain.invoke(input_data)

        if self.memory:
            self.memory.save_context(
                inputs={"question": query},
                outputs={"response": response}
            )

        return response

        
