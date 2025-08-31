import os
import re
import logging
import uuid
import json
import sqlite3
import hashlib
import requests
import textwrap
from typing import List, Dict, Optional, Union,Literal
from io import BytesIO
import tempfile
from typing import Any
import subprocess
from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

import yaml
from PIL import Image
import chromadb
from pydantic import Field, BaseModel
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from rich.syntax import Syntax
from rich.console import Console
from rich.panel import Panel
from sklearn.metrics.pairwise import cosine_similarity

from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser,JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.tools import BaseTool
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever,HumanMessage
from langchain_tavily import TavilySearch

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.documents.elements import CompositeElement,Element
from unstructured.chunking.title import chunk_by_title

from models.Model import Chatbot

import base64


# Solution :https://bennycheung.github.io/ask-a-book-questions-with-langchain-openai 
def split_chuncks(text:List[CompositeElement],filename:str) -> List[Document]: # Solution : https://python.langchain.com/v0.2/docs/tutorials/retrievers/#documents 
    """
    Split the text into chunks and adds metadata to each chunk to link it to the following chunk
    
    params
    ------
    - text: List[CompositeElement]

    returns
    -------
    documents = List[Document]
    """
    page_docs = [Document(page_content=page.text) for page in text]
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

## The primary problem with this approach is that it works quiet well for files that coming from
## locally saved places but still won't work with online pdfs 
def get_pdf_title(pdf_path) -> str:
    """
    Return the title of the pdf file

    params
    ------ 
    - pdf_path (str): Path to the PDF file
    
    returns
    -------
    str: Title of the pdf file if it exists else "No title found"
    """

    pdf_stream = BytesIO(pdf_path)
    reader = PdfReader(pdf_stream)
    metadata = reader.metadata  # Extract metadata
    title = metadata.get('/Title', None)  # Get title if available
    return title or "No title found"


def unload_model(logger, model_name:str=None, base_url = "http://localhost:11434"): # Solution : https://github.com/ollama/ollama/issues/1600
    """
    Unload the given model from memory by calling the Ollama API.

    params
    ------
    - logger (Logger): Logger object for logging messages
    - model_name (str): Name of the model to be unloaded
    - base_url (str): Base URL of the Ollama server (default: "http://localhost:11434")
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


def load_ai_template(config_path: str) -> Dict:
    """
    Load the template present in the template.yaml file

    params
    ------
    - template_name (str): Name of the template to be loaded

    returns
    -------
    Dict: Template configuration loaded from the template.yaml file

    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_file_hash(file: Union[str, BytesIO]) -> str:
    """
    Create the hash for the first 100 bytes of a given file,
    supports local file path, URL, or file-like object.

    params
    ------
    - file (str or file-like): File path, URL, or file-like object

    returns
    -------
    str: MD5 hash of the first 100 bytes of the file
    """
    # Get first 100 bytes depending on input type
    if isinstance(file, str):
        if file.startswith("http://") or file.startswith("https://"):
            # URL: download first 100 bytes
            response = requests.get(file, stream=True)
            response.raise_for_status()
            first_100_bytes = response.raw.read(100)
        else:
            # File path: open and read first 25 bytes
            with open(file, "rb") as f:
                first_100_bytes = f.read(100)
    else:
        # File-like object
        file.seek(0)
        first_100_bytes = file.read(100)
        file.seek(0)  # Reset pointer after reading

    # Hash the first 100 bytes
    file_hash = hashlib.md5(first_100_bytes).hexdigest()
    return file_hash

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines)) 

def get_parent_id(element) -> str:
    """
    Retrieve the parent ID if available, otherwise use its own element ID.
    
    params
    ------
    - element(Dict): Dict containing the metadata from which we retrieve the parent id or the element id 

    returns
    -------
    str: Parent id of the given element or it's own element
    """
    return element["metadata"].get("parent_id", element["element_id"])

# Solution : https://github.com/langchain-ai/langchain/issues/6046 , https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_execute_retrievalqa_chain.py 
class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Optional[Any] = Field(default=None)
    docstore: Optional[Any] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
       
        Retrieve the top 5 most relevant documents for a given query, enriching metadata with scores and optional raw content.
            
        params
        ------
        - query (str): The search query string used to perform similarity matching against the vectorstore.

        returns
        -------
        List[Document]: A list of up to 5 Document objects, each enriched with a 'score' in metadata and optional 'data' from the docstore if available.
        """
        # Perform the similarity search with scores
        docs, scores = zip(*self.vectorstore.similarity_search_with_relevance_scores(query, k=10, score_threshold=0.19))

        # Sort and pick top 5
        docs_with_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:5]

        enriched_docs = []
        for doc, score in docs_with_scores:
            doc.metadata["score"] = score

            doc_id = doc.metadata.get("doc_id")
            if doc_id and self.docstore:
                data = self.docstore.mget([doc_id])[0]
                if data:
                    doc.metadata["data"] = data  # could be table, image_paths.

            enriched_docs.append(doc)

        return enriched_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

class TemporaryDB:
    def __init__(self, db_name: str = "default", use_memory: bool = True) -> None:
        """
        Initialize a temporary or in-memory SQLite database to store key-value pairs.

        params
        ------
        - db_name (str): Name of the database file if not using in-memory storage.
        - use_memory (bool): Whether to use an in-memory database. Defaults to True.
        """
        self.db_name = db_name

        if use_memory:
            self.conn = sqlite3.connect(':memory:')
        else:
            db_filename = f"temp_db_{self.db_name}.sqlite"
            db_path = os.path.join(tempfile.gettempdir(), db_filename)
            self.conn = sqlite3.connect(db_path)

        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS saved_values (
                id TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        self.conn.commit()

    def add(self, uid: str, value: str) -> bool:
        """
        Adds a (UUID, value) pair if it does not already exist.

        params
        ------
        - uid (str): Unique identifier for the value.
        - value (str): The value to store.

        returns
        -------
        bool: True if the value was added, False if the UID already exists.
        """
        if self.exists_by_id(uid):
            return False
        self.cursor.execute("INSERT INTO saved_values (id, value) VALUES (?, ?)", (uid, value))
        self.conn.commit()
        return True

    def exists_by_id(self, uid: str) -> bool:
        """
        Check if a value exists in the database by its unique ID.

        params
        ------
        - uid (str): Unique identifier to check.

        returns
        -------
        bool: True if the ID exists, False otherwise.
        """
        self.cursor.execute("SELECT 1 FROM saved_values WHERE id = ?", (uid,))
        return self.cursor.fetchone() is not None

    def get_by_id(self, uid: str) -> str:
        """
        Retrieve the value associated with a given unique ID.

        params
        ------
        - uid (str): Unique identifier for the value.

        returns
        -------
        str: The stored value if found, None otherwise.
        """
        self.cursor.execute("SELECT value FROM saved_values WHERE id = ?", (uid,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def get_all(self) -> List:
        """
        Retrieve all stored key-value pairs from the database.

        params
        ------
        None

        returns
        -------
        List: A list of tuples [(id, value), ...] representing all entries.
        """
        self.cursor.execute("SELECT id, value FROM saved_values")
        return self.cursor.fetchall()

    def remove_by_id(self, uid: str):
        """
        Remove a value from the database by its unique ID.

        params
        ------
        - uid (str): Unique identifier for the value to remove.
        """
        self.cursor.execute("DELETE FROM saved_values WHERE id = ?", (uid,))
        self.conn.commit()

    def __del__(self):
        """
        Close the database connection when the object is destroyed.
        """
        try:
            self.conn.close()
        except Exception:
            pass

#========================================================================================================= Web srapping/search tools =========================================================================================================# 
class CustomSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Useful for answering current or recent questions using Tavily web search."
    web_search_tool: Optional[TavilySearch] = Field(default=None, exclude=True)
    llm: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self,llm:Optional[Any] = None) -> None:
        super().__init__()
        load_dotenv()
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.llm = llm 
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not set in environment variables.")

        # Initialize the Tavily tool
        self.web_search_tool = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=3,
            topic="general", # Perhaps creating a flag in which we can each time we search for advaned topics 
            include_raw_content = True,
            include_images=True,
            include_image_descriptions=True
        )

    def summarize_content(self,result:dict,query:str,max_words:int = 500) -> str:
        """
        Generate a concise summary of content based on a query.

        params
        ------
        - result (dict): Dictionary containing the content to summarize (expects 'raw_content' key).
        - query (str): The search query or topic the summary should focus on.
        - max_words (int): Maximum number of words allowed in the summary. Defaults to 500.

        returns
        -------
        str: A summary of the content relevant to the query, or a message indicating no content is available.
        """
        summarize_template = """
        You are an expert summarizer.
        Summarize the following content into a clear, accurate, and concise summary, highlighting its relevance to the query.: {query}

        Content:
        {content}

        Guidelines:
        - Keep the summary under {max_words} words.
        - Capture the key points, main ideas, and essential details.
        - Use plain, clear language.
        - Do not include personal opinions or interpretations.
        - Maintain factual accuracy.

        Summary:
        """
        summary_prompt = PromptTemplate(
            input_variables=["content", "max_words"],
            template=summarize_template
        )

        content = result.get("raw_content", "")

        result = None
        if content != "":
            result = self.llm.invoke(summary_prompt.format(content = content, query=query,max_words = max_words)).content
        else:
            result = "No content available for this search result"

        return result

    def format_tavily_response(self, raw_response: Dict,query:str) -> Dict:
        """
        Format a raw web search response for LLM consumption, filtering and summarizing results.

        params
        ------
        - raw_response (Dict): Raw response from a web search, including results and images.
        - query (str): The search query used to guide content summarization.

        returns
        -------
        Dict: Formatted response containing:
            - 'results': List of dicts with 'title', 'url', and summarized 'content'.
            - 'images': List of up to 2 dicts with 'url' and 'description' of images.
        """
        # Filter results with score > 0.60
        filtered_results = [
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "content": self.summarize_content(result,query=query)
            }
            for result in raw_response.get("results", [])
            if result.get("score", 0) > 0.60
        ]

        # Get only the first 2 images with url and description
        images = raw_response.get("images", [])[:2]
        formatted_images = [
            {"url": img.get("url"), "description": img.get("description")}
            for img in images
        ]

        return {
            "results": filtered_results,
            "images": formatted_images
        }
    
    def _run(self, query: Union[str, List[str]]):
        """
        Execute a web search for a single query or a list of queries and return formatted results.

        params
        ------
        - query (str | List[str]): A single search query string or a list of query strings.

        returns
        -------
        dict | list: Formatted search results. Returns a dict for a single query, or a list of dicts for multiple queries.
        """
        results = []
        if isinstance(query, list):
            for single_query in query:
                search_results = self.web_search_tool.invoke({"query": single_query})
                formatted_search_results = self.format_tavily_response(search_results,query=single_query)
                # print(f"search results in tool call: {formatted_search_results}")
                results.append(formatted_search_results)
        else:
            search_results = self.web_search_tool.invoke({"query": query})
            # print(f"search results: {search_results}")
            formatted_search_results = self.format_tavily_response(search_results)
            results = formatted_search_results

        return results

class WebscrapperTool(BaseTool):
    name :str = "web_scrape"
    description:str = "Useful for information scrapping for websites"
    llm: Optional[Any] = Field(default=None, exclude=True)
    embeddings: Optional[OllamaEmbeddings] = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True

    def __init__(self,llm:Chatbot) -> None:
        super().__init__()
        self.llm = llm
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest",
                                                base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
                                            )

    # Solution : https://github.com/Unstructured-IO/unstructured/issues/3642 
    def parse_html(self, url: str = None) -> List[Element]:
        """
        Parse HTML from the provided URL and extract all structured content
        including titles, subtitles, and text elements using Unstructured's partition_html.
        
        params
        ------
        url(str): The URL of the website to scrape.
            
        returns
        -------
        List[Element] A list of unstructured elements (Title, NarrativeText, ListItem, etc.)
            extracted from the HTML content, preserving document structure.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Referer": "https://www.google.com",
                "Connection": "keep-alive",
            }
            session = requests.Session()
            session.headers.update(headers)
            response = session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            
            # More comprehensive content extraction strategy
            main_content = self._extract_main_content(soup)
            
            # Remove unwanted tags but preserve structure
            self._clean_content(main_content)

            main_title = self.extract_main_title(main_content)
            
            # Convert cleaned content to HTML string
            cleaned_html = str(main_content)
            
            # Partition the HTML into unstructured elements with better options
            elements = partition_html(
                text=cleaned_html,
                skip_headers_and_footers=True,
                include_metadata=True,
                chunking_strategy="by_title",
                include_page_breaks=True,
                languages=["en"],  # Adjust as needed
            )
            
            # Return ALL elements to preserve structure (titles, subtitles, text, etc.)
            return elements,main_title
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with web: {e}")
        
    def extract_main_title(self, content: BeautifulSoup) -> str:
        """
        Extracts the first <h1> tag inside the main content.
        
        params
        ------
        content(BeautifulSoup): The main content area of the page.

        returns
        -------
        str The text of the first <h1> found, or an empty string if not present.
        """
        h1_tag = content.find("h1")
        return h1_tag.get_text(strip=True) if h1_tag else ""

    def _extract_main_content(self, soup):
        """
        Extract the main content from an HTML soup using semantic tags, CMS patterns, and fallback strategies.

        params
        ------
        - soup (BeautifulSoup): Parsed HTML content.

        returns
        -------
        Tag: The HTML element containing the main content, or the best fallback available.
        """
        content_selectors = [
            # Semantic HTML5 tags
            "main",
            "article", 
            "[role='main']",
            
            # Common CMS patterns
            ".content",
            ".main-content", 
            ".article-content",
            ".post-content",
            ".entry-content",
            
            # News/blog specific
            ".article-body",
            ".story-body", 
            ".post-body",
            
            # Generic patterns
            "#content",
            "#main",
            "#main-content",
            ".container .content",
        ]
        
        # Try each selector
        for selector in content_selectors:
            try:
                content = soup.select_one(selector)
                if content and len(content.get_text(strip=True)) > 250:  # Minimum content threshold
                    return content
            except:
                continue
        
        # Fallback: find the div with most text content
        divs = soup.find_all("div")
        if divs:
            main_content = max(
                divs,
                key=lambda div: len(div.get_text(strip=True)),
                default=soup
            )
            # Only use if it has substantial content
            if len(main_content.get_text(strip=True)) > 50:
                return main_content
        
        # Final fallback: use body or entire soup
        return soup.find("body") or soup

    def _clean_content(self, content) -> None:
        """
        Remove noise elements such as ads, scripts, headers, footers, and comments while preserving semantic content.

        params
        ------
        - content (Tag): HTML content element to clean.
        """

        # Tags to completely remove
        noise_tags = [
            "script", "style", "noscript",
            "iframe", "embed", "object",
            # Navigation and UI elements
            "nav", "header", "footer", 
            # Ads and social
            ".advertisement", ".ad", ".ads",
            ".social-share", ".share-buttons",
            # Comments and related
            ".comments", ".comment-section",
            ".related-articles", ".sidebar",
            # Cookie/privacy notices
            ".cookie-notice", ".privacy-notice"
        ]
        
        for selector in noise_tags:
            if selector.startswith('.'):
                # CSS class selector
                for tag in content.select(selector):
                    tag.decompose()
            else:
                # Tag name
                for tag in content.find_all(selector):
                    tag.decompose()
    

    def summarize_text(self, title, text) -> str:
        """
        Summarize a given text section with respect to its title using an LLM.

        params
        ------
        - title (str): Section title providing context for the summary.
        - text (str): Text content to summarize.

        returns
        -------
        str: The generated summary of the text section.
        """
        prompt = PromptTemplate(
            template=f"Given the section title: '{title}', summarize the following text:\n{text}",
            input_variables=["text","title"]
        )

        return self.llm.invoke(prompt.format(text=text, title=title))

    def identify_relevant_passages(self, elements: List[Element], keywords: List[str], threshold: float = 0.5) -> List[Element]:
        """
        Identify elements whose content is semantically relevant to a set of keywords.

        params
        ------
        - elements (List[Element]): List of HTML elements or text sections to evaluate.
        - keywords (List[str]): Keywords to match against content.
        - threshold (float): Minimum cosine similarity required for relevance. Defaults to 0.5.

        returns
        -------
        List[Element]: Subset of elements deemed relevant to the keywords.
        """
        if not keywords:
            return []

        # Embed all keywords as one string or individually
        keyword_text = " ".join(keywords)
        keyword_embedding = self.embed_text(keyword_text)

        relevant_elements = []

        for el in elements:
            if not hasattr(el, 'text') or not el.text.strip():
                continue

            element_embedding = self.embed_text(el.text)
            sim = cosine_similarity([keyword_embedding], [element_embedding])[0][0]

            if sim >= threshold:
                relevant_elements.append(el)

        return relevant_elements
    
    def embed_text(self, text: str) -> List[float]:
        """
        Convert text into a numerical embedding vector using the configured embedding model.

        params
        ------
        - text (str): Text to embed.

        returns
        -------
        List[float]: Embedding vector representing the input text.
        """
        return self.embeddings.embed_query(text)

    def chunk_large_text(self, title:str, elements: List, max_tokens: int = 1024) -> List[str]:
        """
        Split a large text into smaller chunks for easier processing or summarization.

        params
        ------
        - title (str): Section title for context.
        - elements (List): List of text-containing elements to combine and split.
        - max_tokens (int): Maximum tokens per chunk. Defaults to 1024.

        returns
        -------
        List[Dict]: List of dictionaries containing 'title' and 'text' for each chunk.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=50
        )
        text = " ".join([el.text for el in elements])# Combine section text
        chunks = splitter.split_text(text)  # Split into manageable parts
        return [{"title": title, "text": chunk} for chunk in chunks]
    
    def generate_summaries(self, relevant_sections) -> List[Dict]:
        """
        Generate summaries for a list of relevant sections, optionally chunking large sections.

        params
        ------
        - relevant_sections (List[Dict]): List of sections, each with 'title' and 'elements'.

        returns
        -------
        List[Dict]: List of summaries with keys 'title' and 'summary' for each section.
        """
        summaries = []
        title = ""
        section_summary = ""
        for section in relevant_sections:
            title = section["title"]
            elements = section["elements"]
            
            # Chunk the section
            chunks = self.chunk_large_text(title, elements)

            # Summarize each chunk and combine
            section_summary = " ".join(self.summarize_text(chunk["title"], chunk["text"]).content for chunk in chunks)

        summaries = {"title": title, "summary": section_summary}
        
        return summaries

    def _run(self, keywords: Union[str], url: str) -> List[Dict]:
        """
        Parse HTML from a URL, extract content, identify relevant passages by keywords, and summarize.

        params
        ------
        - keywords (str | List[str]): Keywords to filter relevant passages. If None, summarizes all content.
        - url (str): URL of the web page to process.

        returns
        -------
        List[Dict]: List of relevant sections or summarized content, structured as dictionaries.
        """
        elements,title = self.parse_html(url=url)
        keywords = [keywords] if isinstance(keywords, str) else (keywords if isinstance(keywords, list) and keywords else None)

        if keywords:
            # Now returns structured list of sections
            relevant_sections = self.identify_relevant_passages(elements, keywords)
            return relevant_sections
        else:
            # Summarize all content if no keywords are given
            full_content_section = {"title": title, "elements": elements}
            return self.generate_summaries([full_content_section])
    
    def _arun(self):
        raise NotImplementedError("This tool does not support async")


#========================================================================================================= Code Tools =========================================================================================================#
def format_code(code: str) -> str:
    # return f"```{lang}\n{textwrap.dedent(code)}\n```"
    return f"\n{textwrap.dedent(code)}\n"

class CodeGeneratorTool(BaseTool):
    name: str = "code_tool"
    description: str = "Handles code-related tasks such as syntax highlighting, code analysis, and debugging."
    llm: Optional[Chatbot] = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True

    def __init__(self,llm:Chatbot) -> None:
        super().__init__()
        self.llm = llm

    def _run(self, query: str, steps: List[str],chat_history:str) -> dict:
        """
        Uses an LLM to generate Python code based on the query and structured steps.

        params
        ------
        - query (str): The programming task description.
        - steps (List[str]): A structured sequence of steps to implement the solution.
        - chat_history(str): THe previously generated code history 

        returns
        -------
        str: The generated Python code.
        """
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        prompt = self._generate_prompt()
        llm_chain = prompt | self.llm | StrOutputParser()
        response = llm_chain.invoke({"query": query, "steps_text": steps_text, "chat_history": chat_history})
        return response

    def _generate_prompt(self) -> PromptTemplate:
        """
        Formats the query and steps into a structured prompt for the LLM.

        returns
        -------
        str: A formatted prompt.
        """
        
        try:
            templates = load_ai_template(config_path="config/template.yaml")
            template_config = templates["Agent_templates"]["Code_generator_template"]
            template = template_config["template"]
            input_variables = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize code research template: {str(e)}")
    
    def _arun(self):
        raise NotImplementedError("This tool does not support async")

class CodeCorrectorTool(BaseTool):
    name: str = "code_corrector_tool"
    description: str = "Corrects syntax errors, logical mistakes, and formatting issues in Python code."
    llm: Optional[Chatbot] = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: Chatbot) -> None:
        super().__init__()
        self.llm = llm

    def _run(self,code,feedbacks:List[Dict]) -> str:
        """
        Apply major feedback corrections to a code snippet using an LLM.

        params
        ------
        - code (str): The source code to correct.
        - feedbacks (List[Dict]): List of feedback dictionaries containing 'issue', 'recommendation', and 'severity'.

        returns
        -------
        str: Corrected code generated by the LLM based on the major feedback steps.
        """
        steps_text = ""
        for feedback in feedbacks:
            if feedback.get("severity") == "Major":
                # Properly format the issue and recommendation as steps
                steps_text += f"\n- Issue: {feedback.get('issue')}\n  Recommendation: {feedback.get('recommendation')}\n"
        prompt = self._generate_correction_prompt()
        llm_chain = prompt | self.llm | StrOutputParser()
        # Call the LLM
        response = llm_chain.invoke({"code": code, "feedback": steps_text})
        return response

    def _generate_correction_prompt(self) -> PromptTemplate:
        """
        Load and construct a code correction prompt template from configuration.

        returns
        -------
        PromptTemplate: A PromptTemplate object ready to be used with the LLM for code correction.

        raises
        ------
        ValueError: If the template cannot be loaded or initialized.
        """
        try:
            templates = load_ai_template(config_path="config/template.yaml")
            template_config = templates["Agent_templates"]["Code_correction_template"]
            template = template_config["template"]
            input_variables = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize code correction template: {str(e)}")

class CodeReviewTool(BaseTool):
    name: str = "code_review_tool"
    description: str = "Analyzes and reviews code snippets for potential bugs, errors, and performance issues."
    llm: Optional[Chatbot] = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: Chatbot):
        super().__init__()
        self.llm = llm

    def _run(self, code: str) -> Dict:
        """
        Analyzes the provided code and returns a structured critique.

        params
        ------
        - code (str): The generated Python code to review.

        returns
        -------
        Dict: A JSON response containing:
            - critique_feedback (list): Detailed areas of improvement.
            - is_code_valid (bool): Whether the code is production-ready.
        """
        # Generate the LLM prompt
        prompt = self._generate_review_prompt()
        llm_chain = prompt | self.llm | JsonOutputParser()
        # Call the LLM
        response = llm_chain.invoke({"code":code})
        return response

    def _generate_review_prompt(self) -> PromptTemplate:
        """
        Generates a structured prompt for code review.

        params
        ------
        - code(str): The code snippet to be reviewed.

        returns
        -------
        str: A structured prompt for the LLM.
        """
        try:
            templates = load_ai_template(config_path="config/template.yaml")
            template_config = templates["Agent_templates"]["Code_review_tempalte"]
            template = template_config["template"]
            input_variables = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize code research template: {str(e)}")


    def _arun(self):
        raise NotImplementedError("This tool does not support async")

class CodeDiagramCreator(BaseTool):
    name: str = "code_diagram_creator"
    description: str = "Creates diagrams from code snippets."
    llm: Optional[Chatbot] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: Chatbot):
        super().__init__()
        self.llm = llm
    
    def _run(self, query: str) -> Dict:
        """
        Generate a diagram or structured representation based on a given code snippet or query.

        params
        ------
        - query (str): The code snippet or textual description used to generate the diagram.

        returns
        -------
        Dict: The structured output or diagram generated by the LLM, parsed as JSON.
        """
        prompt = self._generate_review_prompt()
        llm_chain = prompt | self.llm | JsonOutputParser()

        response = llm_chain.invoke({"query": query})
        return response
        
    def _generate_review_prompt(self) -> PromptTemplate:
        """
        Generates a structured prompt for code review.

        params:
        -------
            code (str): The code snippet to be reviewed.

        returns:
        --------
            str: A structured prompt for the LLM.
        """
        try:
            templates = load_ai_template(config_path="config/template.yaml")
            template_config = templates["Agent_templates"]["Code_diagram_template"]
            template = template_config["template"]
            input_variables = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize code research template: {str(e)}")
    

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

#========================================================================================================= Other functions =========================================================================================================#

def parse_flags_and_queries(input_text: str) -> dict[str, str]:
    """
    Parses the input text to retrieve all flags and their corresponding queries.
    Flags can appear anywhere in the input, and each flag's query continues until
    another flag is found or the input ends.

    params
    ------
        input_text (str): The raw input string from the user.

    returns
    -------
        dict: A dictionary where keys are flags (e.g., '/code') and values are the queries.
    """
    input_text = input_text.strip()
    
    # Match a flag and capture everything until the next flag or end of string
    pattern = r"(\/\w+)\s+(.*?)(?=\s+\/\w+|$)"
    
    matches = re.findall(pattern, input_text, re.DOTALL)
    
    flag_query_dict = {}
    
    for flag, query in matches:
        flag_query_dict[flag] = query.strip()
    
    return flag_query_dict

def pretty_print_query(query,console:Console) -> None:
    """
    Prints the user query in a formatted panel.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    panel = Panel.fit(
        query,
        title=f"🧑 You [{timestamp}]",
        border_style="bold magenta",
        padding=(1, 2)
    )
    console.print(panel)

def pretty_print_answer(answer,console:Console) -> None:
    """
    Prints the given answer in a formatted panel using the 'rich' library.

    params
    ------
        - answer (str): The answer to be printed.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    panel = Panel.fit(
        answer,
        title=f"🤖 Bot [{timestamp}]",
        border_style="bold cyan",
        padding=(1, 2)
    )
    console.print(panel)

def pretty_print_code(code: str, console: Console, language: str = "python") -> None:
    """
    Pretty print code with syntax highlighting.

    params
    ------
    - code(str): The source code to display.
    - console(console): Rich Console instance for output.
    - language(str): Programming language for syntax highlighting (default is 'python').
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)

def detect_language_lib(code: str) -> str:
    try:
        lexer = guess_lexer(code)
        return lexer.name.lower()
    except ClassNotFound:
        return None

def deduplicate(docs: List[Union[Document, str]], k: int = 10) -> List[Union[Document, str]]:
    """
    Deduplicates retrieved documents (instances of Document), keeping only the highest-scoring version of each,
    and selects the top `k` documents. In addition, any strings present in the input are returned unmodified.

    params
    ------
        - docs (List[Union[Document, str]]): List of retrieved items which may include Documents with metadata (including scores)
                                            and strings.
        - k (int): Number of top-scoring unique Document objects to return. Default is 10.

    returns
    -------
    List[Union[Document, str]]: A list containing the top `k` unique Documents (sorted by highest score) 
                                    plus any strings that were present in the original list.
    """
    # Separate Document instances and strings
    document_items = []
    string_items = []
    for item in docs:
        if isinstance(item, Document):
            document_items.append(item)
        elif isinstance(item, str):
            string_items.append(item)
        else:
            pass

    doc_score_map = defaultdict(lambda: float('-inf'))  # Maps doc_id to the highest score seen.
    doc_map = {}  # Maps doc_id to the best Document instance.
    for doc in document_items:
        # Use metadata 'doc_id' if available, otherwise fall back to doc.id
        doc_id = doc.metadata.get("doc_id", getattr(doc, "id", None))
        if doc_id is None:
            continue  # or decide to include these items without deduplication

        score = doc.metadata.get("score", 0)
        if score > doc_score_map[doc_id]:
            doc_score_map[doc_id] = score
            doc_map[doc_id] = doc

    # Get top `k` highest-scoring unique Documents
    sorted_docs = sorted(doc_map.values(), key=lambda d: d.metadata.get("score", 0), reverse=True)[:k]

    # Return both the deduplicated/sorted Document objects and the strings.
    return sorted_docs + string_items
    
def get_ollama_model_details(model_name: str, url:str = "http://localhost:11434/api/show"):
    """
    Retrieve detailed information about a specified Ollama model from the local API.

    params
    ------
    - model_name (str): Name of the Ollama model to query.
    - url (str): URL of the Ollama API endpoint. Defaults to "http://localhost:11434/api/show".

    returns
    -------
    Dict | None: Dictionary containing the model's architecture, parameter size, context length,
                 embedding length, and quantization level. Returns None if the request fails
                 or the response cannot be parsed.
    """

    payload = {"model": model_name}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise error if request fails
        
        data = response.json()  # Parse JSON response
        
        # Extract required parameters
        model_details = {
            "architecture": data["model_info"].get("general.architecture", "N/A"),
            "parameters": f'{data["details"].get("parameter_size", "N/A")}',
            "context_length": data["model_info"].get("llama.context_length", "N/A"),
            "embedding_length": data["model_info"].get("llama.embedding_length", "N/A"),
            "quantization": data["details"].get("quantization_level", "N/A")
        }
        
        return model_details

    except requests.exceptions.RequestException as e:
        logging.exception("HTTP Request failed:", e)
        return None
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON response.")
        return None

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

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, NAME: str = None) -> None:
        """
        Initialize a persistent vector database client with embeddings and logging.

        params
        ------
        - NAME (str): Optional name for the Chroma collection. Defaults to "knowledge_base" if None.

        notes
        -----
        - Sets up Ollama embeddings with a local server.
        - Ensures the database directory exists.
        - Initializes a persistent Chroma client and collection.
        - Sets up a vector store for embedding-based operations.
        - Initializes logging.
        - Skips re-initialization if already initialized.
        """
        if hasattr(self, 'initialized'):
            return  # Already initialized, skip re-init

        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434"
        )

        db_path = "database/"
        os.makedirs(db_path, exist_ok=True)  # Make sure dir exists

        self.client = chromadb.PersistentClient(path=db_path)

        collection_name = "knowledge_base" if NAME is None else NAME # This defines the collection default name as knowledge_base 

        # Try to get the collection, create only if missing
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

        self.logger = self._setup_logging()
        self.initialized = True

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def populate_vector(self, documents: List[Document]) -> bool:
        """
        Populates the vectordb with the embedding of the given file

        params
        ------
            - documents (List[Document]): List of documents to be added to the vector store

        returns
        -------
        bool: True if all documents are added to the vector store successfully, False otherwise
        """
        
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

        params
        ------
            - query (str): Query to be searched in the knowledge base
            - filter (Dict): Additional filter parameters for the search
            - score (bool): Whether to return the score along with the documents (default: False)
            - k (int): Number of documents to return (default: 2)

        returns
        -------
        List[Document]: List of documents that match the search query, up to 'k' documents.
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

#========================================================================================================= PDF Tools =========================================================================================================#
class PDF_Reader:
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

    def read_pdf(self, filename: Union[BytesIO,str]):
        """
        Read and partition a PDF file into chunks. It extracts tables, images, and text with specific chunking strategies.

        params
        ------
            filename Union[BytesIO,str]: A BytesIO object containing the PDF file to be read and partitioned.

        returns
        -------
            List (CompositeElement or None): A list of CompositeElement objects representing the partitioned PDF content. 
            Returns None if an error occurs during the partitioning process.

        raises
        ------
            AssertionError : If the input file is not a PDF (doesn't end with '.pdf').

        notes
        -----
        The function uses a temporary file to store the PDF content before processing.
        The temporary file is deleted after the partitioning is complete.
        """
        if isinstance(filename,str):
            assert filename.endswith('.pdf'), "Given file should be a .pdf"
            temp_filename = filename

        elif isinstance(filename, BytesIO):
            assert filename.name.endswith('.pdf'), "Given file should be a .pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(filename.read())
                    temp_filename = temp_file.name
        
        try:
            chunks = partition_pdf(
                filename=temp_filename,
                infer_table_structure=True,            # extract tables
                strategy="hi_res",                     # mandatory to infer tables

                extract_image_block_types=["Image", "Table"],   
                extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

                chunking_strategy="by_title",          
                max_characters=10000,                  
                combine_text_under_n_chars=2000,       
                new_after_n_chars=6000,

            )
            filename = self.get_pdf_title(chunks)
            for element in chunks:
                element.metadata.filename = filename

            if isinstance(filename,BytesIO):
                os.remove(temp_filename)
                
            return chunks

        except Exception as e:
            self.logger.error("Failed to partition PDF file %s", e)
            return None
        
    def read_pdf_online(self, url: str):
        """
        Read and process a PDF or HTML page from a URL, extracting text, tables, and images.

        params
        ------
        - url (str): URL pointing to a PDF file or a web page.

        returns
        -------
        List[CompositeElement] | None: List of processed content chunks with metadata including filename.
                                    Returns None if processing fails.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        # First verify if the url is a pdf or webpage
        if url.endswith(".pdf"):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                
                # Extract with unstructured
                elements = partition_pdf(
                    filename=tmp_path,
                    infer_table_structure=True,            # extract tables
                    strategy="hi_res",                     # mandatory to infer tables

                    extract_image_block_types=["Image", "Table"],   
                    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage   
                    max_characters=10000,                  
                    combine_text_under_n_chars=2000,       
                    new_after_n_chars=6000,
                )
                
                chunks = chunk_by_title(elements)
                filename = self.get_pdf_title(chunks)
                for element in chunks:
                    element.metadata.filename = filename
                    
                os.unlink(tmp_path)
                return chunks
                
            except Exception as e:
                print(f"Error processing direct PDF: {e}")
                return None
        
        else:
            # Handle webpage formats (MDPI,) probably add also research gate but having access error 404 forbidden url problems: 
            # Solution for such can be resolved using this : https://stackoverflow.com/questions/72347165/python-requests-403-forbidden-error-while-downlaoding-pdf-file-from-www-resear 
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "lxml")

                main_content = None
                if soup.find("main"):
                    main_content = soup.find("main")
                elif soup.find("article"):
                    main_content = soup.find("article")
                elif soup.find("div", {"id": "content"}):
                    main_content = soup.find("div", {"id": "content"})
                elif soup.find("div", {"id": "main-content"}):
                    main_content = soup.find("div", {"id": "main-content"})
                else:
                    # Fallback: take the largest <div> with the most text
                    main_content = max(soup.find_all("div"), key=lambda div: len(div.text), default=soup)

                # Remove unwanted elements (e.g., navigation, footer, ads)
                for tag in main_content.find_all(["header", "footer", "nav", "aside", "script", "style"]):
                    tag.extract()
                
                cleaned_html = str(main_content)
                
                filename = main_content.h1.text if main_content.h1 else "default_filename"
                filename = ' '.join(filename.split())
                
                elements = partition_html(text=cleaned_html)
                chunks = chunk_by_title(elements)
                for element in chunks:
                    element.metadata.filename = filename
                
                return chunks
            except Exception as e:
                print(f"Error processing webpage: {e}")
                return None
    
    def view_pdf_image(self,base_string:str):
        """
        Display an image extracted from a base64-encoded string, if it represents an image.

        params
        ------
        - base_string (str): Base64-encoded image data.

        returns
        -------
        None
        """
        try:
            if self.is_image_data(base_string):
                image_bytes = self.base64_to_image(base_string)
                image = self.create_image_from_bytes(image_bytes)
                image.show()
        except Exception as e:
            self.logger.error("Failed to view PDF image %s", e)

    @staticmethod
    def base64_to_image(base64_string:str):
        """
        Convert a base64-encoded string into raw image bytes.

        params
        ------
        - base64_string (str): Base64-encoded image data.

        returns
        -------
        bytes: Decoded image data.
        """
        # Remove the data URI prefix if present
        if "data:image" in base64_string:
            base64_string = base64_string.split(",")[1]

        # Decode the Base64 string into bytes
        image_bytes = base64.b64decode(base64_string)
        return image_bytes
    
    @staticmethod
    def create_image_from_bytes(image_bytes:bytes):
        """
        Create a PIL Image object from raw image bytes.

        params
        ------
        - image_bytes (bytes): Raw image data.

        returns
        -------
        Image: PIL Image object.
        """
        # Create a BytesIO object to handle the image data
        image_stream = BytesIO(image_bytes)

        # Open the image using Pillow (PIL)
        image = Image.open(image_stream)
        return image
    
    @staticmethod
    def is_image_data(b64data): 
        """
        Check if a base64 string represents an image based on its file signature. This solution is based from : https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb 

        params
        ------
        - b64data (str): Base64-encoded data.

        returns
        -------
        bool: True if the data represents a recognized image format, False otherwise.
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
        """
        Separate a list of content chunks into text elements, tables, and images.

        params
        ------
        - chunks (List[CompositeElement]): List of content chunks extracted from PDF or HTML.

        returns
        -------
        Tuple[List, List[str], List[str]]: Lists of text elements, table HTML strings, and base64 images.
        """
        tables = []
        texts = []

        for chunk in chunks:
            if "CompositeElement" in str(type((chunk))):
                texts.append(chunk)

        images_b64 = []
        for chunk in chunks:
            # if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
            for el in chunk_els:
                if "Table" in str(type(el)):
                    tables.append(el.metadata.text_as_html)
        return texts,tables,images_b64

    def _get_summaries_table(self,tables:List[str], chatbot:Chatbot):
        """
        Generate summaries for extracted tables using a chatbot.

        params
        ------
        - tables (List[str]): List of table HTML strings.
        - chatbot (Chatbot): Chatbot instance to perform summarization.

        returns
        -------
        List[str]: Summaries of each table. Returns empty list if summarization fails.
        """
        templates = load_ai_template("config/template.yaml")
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
        Generate summaries for extracted images using a chatbot.

        params
        ------
        - images (List[str]): List of base64-encoded image strings.
        - chatbot (Chatbot): Chatbot instance to perform summarization.

        returns
        -------
        List[str]: Summaries of each image. Returns empty list if summarization fails.
        """

        templates = load_ai_template("config/template.yaml")
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
    
    def get_pdf_title(self, chunk:List[CompositeElement]):
        """
        Retrieve the title of a PDF from its extracted content chunks.

        params
        ------
        - chunk (List[CompositeElement]): List of content chunks extracted from a PDF.

        returns
        -------
        str: The PDF title inferred from the content. Defaults to first title element if multiple exist.
        """
        len_el = 0
        if(len(chunk[len_el].text)>300):
            element = chunk[len_el].metadata.orig_elements
        else:
            len_el = 1
            element = chunk[len_el].metadata.orig_elements
        chunk_title = [el for el in element if 'Title' in str(type(el))]
        chunk_title[0].to_dict()
        i = 0
        if len(chunk_title) > 3:
            i=2
            return chunk_title[i].text
        else:
            return chunk_title[i].text
         
   