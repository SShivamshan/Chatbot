from PyPDF2 import PdfReader
import logging
import re
from io import BytesIO
import chromadb
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List,Dict,Optional
from langchain_ollama import OllamaEmbeddings
from langchain.tools import BaseTool
from langchain_chroma import Chroma
from langchain_core.runnables import chain
import yaml
from pydantic import Field
import hashlib


# Solution :https://bennycheung.github.io/ask-a-book-questions-with-langchain-openai 

def parse_pdf(file:BytesIO): # Solution : https://github.com/andreaxricci/pdf-GPT/blob/main/utils.py   
    """Parse Pdf file and return """
    try:
        pdf_reader = PdfReader(file)
        output = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            # Merge hyphenated words
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            # Fix newlines in the middle of sentences
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            # Remove multiple newlines
            text = re.sub(r"\n\s*\n", "\n\n", text)
            # Additional cleanup: normalize spaces
            text = re.sub(r"\s+", " ", text)
            output.append(text)
        logging.info(f"Parsing PDF file! {file.name}")
        return output
    except Exception as e:
        logging.error(f"Error parsing PDF file: {e}")
        return None

    
def split_chuncks(text:List[str]) -> List[Document]: # Solution : https://python.langchain.com/v0.2/docs/tutorials/retrievers/#documents 
    """
    Split the text into chunks and adds metadata to each chunk to link it to the following chunk
    
    params
    ------
        text: List[str]

    returns
    -------
       documents = List[Document]
    """
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Adjusted text splitter with larger overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=200,  
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        length_function=len,
    )

    # Split the text into chunks and add metadata
    doc_chunks = []
    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk, 
                metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources as metadata
            chunk_doc.metadata["source"] = f"{chunk_doc.metadata['page']}-{chunk_doc.metadata['chunk']}"
            doc_chunks.append(chunk_doc)

    logging.info("File has been chunked with metadata: %i", len(doc_chunks))
    return doc_chunks

# Adopted solution : https://medium.com/@arunpatidar26/rag-chromadb-ollama-python-guide-for-beginners-30857499d0a0 
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings:OllamaEmbeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)
    
class Vectordb:
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self) -> None:
        if not hasattr(self, 'initialized'):
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest",
                                                base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
                                            )
            self.client = chromadb.PersistentClient(path="database/")
            self.collection = self.client.get_or_create_collection(name="knowledge_base",metadata={"hnsw:space": "cosine"})
            self.vector_store = Chroma(client=self.client,collection_name="knowledge_base",embedding_function=self.embeddings)
            self.initialized = True
    
    def populate_vector(self, documents: List[Document]) -> bool:
        """Populates the vectordb with the embedding of the given file"""
        try:
            existing_count = self.collection.count() # Returns the number of embedded elements in the vector db
            for i, doc in enumerate(documents):
                try:
                    unique_id = f"doc_{existing_count + i}"
                    self.vector_store.add_documents(
                        documents=[doc],
                        ids=[unique_id]
                    )
                except Exception as e:
                    logging.error(f"Error adding document {i} to vector store: {e}")

            logging.info("All documents added to vector store successfully.")
            return True
        except Exception as e:
            logging.error(f"Error populating vector store: {e}")
            return False

    def search_similarity(self,query:str,filter:Dict = None,score:bool = False,k:int = 2) -> List[Document]:
        """
        Search the knowledge base with the given query and return up to 4 documents
        """
        # Search for similarity between documents in the knowledge base and the query
        try:
            if score : 
                results = self.vector_store.similarity_search_with_score(query=query, filter=filter, k=k)
            else:
                results = self.vector_store.similarity_search(query=query,filter=filter,k=k)
            return results
        except Exception as e:
            logging.error(f"Error searching for context: {e}")
            return None
        
    def retrieve_similarity(self,query,filter:Dict = None,k:int = 2):
        
        retriever = self.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": k, "fetch_k": self.collection.count()}
            )

        return retriever.invoke(input=query,kwargs=filter)

class SearchTool(BaseTool):
    name: str = "SearchRelevantInformation"
    description: str = "Search the knowledge base for relevant information."
    vector_db: Optional['Vectordb'] = Field(default=None, exclude=True)

    def __init__(self, retriever:Optional['Vectordb'] = None):
        super().__init__()
        if retriever:
            self.vector_db = retriever  # Assign the passed retriever to vector_db
        else:
            vector_store = Vectordb()
            collection_size = vector_store.collection.count()
            self.vector_db = vector_store.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": collection_size}
            )

    async def _arun(self, query: str) -> str:
        """Run async implementation"""
        raise NotImplementedError("SearchTool does not support async")
    def _run(self, query: str):
        """
        Search for relevant information in the vector database and return the formatted result.
        """
        # Perform the search using the retriever
        response = self.vector_db.invoke(query)
        # Format the results
        formatted_results = self._format_results(response)

        return formatted_results
    @staticmethod
    def _format_results(response: List[tuple]) -> str:
        """
        Format the search results by preparing a structured response without scores.
        """
        # Parse the response and create a structured list of results
        parsed_results = []
        for res in response:
            parsed_results.append({
                "Content": res.page_content,
                "Page": res.metadata.get("page"),
                "Source": res.metadata.get("source"),
            })
        return parsed_results

class CitationTool(BaseTool):
    name: str = "GenerateCitation"
    description: str = "Generate a citation from a relevant section or page in the PDF."
    vector_db: Optional['Vectordb'] = Field(default=None, exclude=True)

    def __init__(self, retriever:Optional['Vectordb'] = None):
        super().__init__()
        if retriever:
            self.vector_db = retriever  # Assign the passed retriever to vector_db
        else:
            vector_store = Vectordb()
            collection_size = vector_store.collection.count()
            self.vector_db = vector_store.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": collection_size}
            )

    async def _arun(self, query: str) -> str:
        """Run async implementation"""
        raise NotImplementedError("SearchTool does not support async")

    def _run(self, query: str):
        """
        Search for relevant information in the vector database and return the formatted result.
        """
        # Perform the search using the retriever
        response = self.vector_db.invoke(query)
        # Format the results
        formatted_results = self._format_results(response)

        return formatted_results
    
    @staticmethod
    def _format_results(response: List[tuple]) -> str:
        """
        Format the search results by preparing a structured response without scores.
        """
        # Parse the response and create a structured list of results
        parsed_results = []
        for res in response:
            parsed_results.append({
                "Content": res.page_content,
                "Page": res.metadata.get("page"),
                "Source": res.metadata.get("source"),
                "Chunk" : res.metadata.get("chunk"),
            })

        return parsed_results

def load_ai_template(template_name: str) -> Dict:
    with open(template_name, 'r') as file:
        config = yaml.safe_load(file)
    return config['ai_templates']


def get_file_hash(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash


