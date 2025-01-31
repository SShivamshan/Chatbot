# Standard library imports
import os
import logging
import uuid
import hashlib
from typing import List, Dict, Optional
from io import BytesIO
import tempfile
from typing import Any
import subprocess

# Third-party library imports
import yaml
from PIL import Image
import chromadb
from pydantic import Field


# LangChain imports
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.tools import BaseTool
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Unstructured library imports
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import CompositeElement

# Local imports
from models.Model import Chatbot

# Utility imports
import base64


# Solution :https://bennycheung.github.io/ask-a-book-questions-with-langchain-openai 

def split_chuncks(text:List[CompositeElement]) -> List[Document]: # Solution : https://python.langchain.com/v0.2/docs/tutorials/retrievers/#documents 
    """
    Split the text into chunks and adds metadata to each chunk to link it to the following chunk
    
    params
    ------
        text: List[CompositeElement]

    returns
    -------
       documents = List[Document]
    """
    page_docs = [Document(page_content=page.text) for page in text]
    filename = text[0].to_dict()["metadata"].get("filename", None)
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
    doc_ids = []
    id_key = "doc_id"
    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            chunk_doc = Document(
                page_content=chunk, 
                metadata={id_key: doc_id ,"page": doc.metadata["page"], "chunk": i, "filename": filename}
            )
            # Add sources as metadata
            chunk_doc.metadata["source"] = f"{chunk_doc.metadata['page']}-{chunk_doc.metadata['chunk']}"
            doc_chunks.append(chunk_doc)
            doc_ids.append(doc_id)

    logging.info("File has been chunked with metadata: %i", len(doc_chunks))
    return doc_chunks,doc_ids

def unload_model(logger, model_name:str=None, base_url = "http://localhost:11434"): # Solution : https://github.com/ollama/ollama/issues/1600 
    """
    Unload the given model from memory by calling the Ollama API.
    """
    curl_command = [
        "curl",
        f"{base_url}/api/generate",
        "-d", f'{{"model": "{model_name}", "keep_alive": 0}}'
    ]
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        # Check if the command was successful
        if result.returncode == 0:
            logger.info(f"Successfully unloaded model: {model_name}")
        else:
            logger.info(f"Failed to unload model: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error unloading model: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")


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
            self.logger = self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)  
        return logger
    
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
                    self.logger.error(f"Error adding document {i} to vector store: {e}")

            self.logger.info("All documents added to vector store successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error populating vector store: {e}")
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
            self.logger.error(f"Error searching for context: {e}")
            return None


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
    return config


def get_file_hash(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash


class PdfReader:
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def read_pdf(self,filename:BytesIO):
        assert filename.name.endswith('.pdf'), "Given file should be a .pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(filename.read())
                temp_filename = temp_file.name
        try:
            chunks = partition_pdf(
                filename=temp_filename,
                infer_table_structure=True,            # extract tables
                strategy="hi_res",                     # mandatory to infer tables

                extract_image_block_types=["Image", "Table"],   # Add 'Table' to list to extract image of tables
                # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

                extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

                chunking_strategy="by_title",          # or 'basic'
                max_characters=10000,                  # defaults to 500
                combine_text_under_n_chars=2000,       # defaults to 0
                new_after_n_chars=6000,

            )
            os.remove(temp_filename)
            return chunks
        
        except Exception as e:
            self.logger.error("Failed to partition PDF file %s", e)
            return None
        
    def view_pdf_image(self,base_string:str):
        try:
            if self.is_image_data(base_string):
                image_bytes = self.base64_to_image(base_string)
                image = self.create_image_from_bytes(image_bytes)
                image.show()
        except Exception as e:
            self.logger.error("Failed to view PDF image %s", e)
    @staticmethod
    def base64_to_image(base64_string:str):
        # Remove the data URI prefix if present
        if "data:image" in base64_string:
            base64_string = base64_string.split(",")[1]

        # Decode the Base64 string into bytes
        image_bytes = base64.b64decode(base64_string)
        return image_bytes
    @staticmethod
    def create_image_from_bytes(image_bytes:bytes):
        # Create a BytesIO object to handle the image data
        image_stream = BytesIO(image_bytes)

        # Open the image using Pillow (PIL)
        image = Image.open(image_stream)
        return image
    @staticmethod
    def is_image_data(b64data): # Solution : https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb 
        """
        Check if the base64 data is an image by looking at the start of the data
        """
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
            for sig, format in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def separate_elements(self,chunks):
        tables = []
        texts = []

        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)

            if "CompositeElement" in str(type((chunk))):
                texts.append(chunk)

        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)


        return texts,tables,images_b64

    def _get_summaries_table(self,tables:List, chatbot:Chatbot):
        """
        Creates the summaries for tables and returns them
        """
        templates = load_ai_template("config/config.yaml")
        table_template_text = templates["Prompt_templates"]["Table_templates"]["template"]
        input_variables = [var["name"] for var in templates["Prompt_templates"]["Table_templates"].get("input_variables", [])]

        table_prompt = ChatPromptTemplate.from_template(
            template=table_template_text,
            input_variable=input_variables
        )

        try:
            summarize_chain = {"element": lambda x: x} | table_prompt | chatbot | StrOutputParser()
            tables_summaries = summarize_chain.batch(tables, {"max_concurrency": 3})

            return tables_summaries
        except Exception as e:
            self.logger.error("Failed to get summaries for tables %s", e)
            return []

    def _get_summaries_image(self,images : List[str] ,chatbot:Chatbot):
        """
        Creates the summaries for images and returns them
        """

        templates = load_ai_template("config/config.yaml")
        image_template_text = templates["Prompt_templates"]["Image_templates"]["template"]

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": image_template_text},
                    {
                        "type": "image",
                        "image_url": {"url": "data:image/jpeg;base64,{images}"},
                    },
                ],
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        try:
            chain = prompt | chatbot | StrOutputParser()
            image_summaries = chain.batch(images)

            return image_summaries
        except Exception as e:
            self.logger.error("Failed to get summaries for images %s", e)
            return []
        
    def get_pdf_title(self,chunk):
        elements = chunk.metadata.orig_elements
        chunk_title = [el for el in elements if 'Title' in str(type(el))]
        chunk_title[0].to_dict()

        return chunk_title[0].text