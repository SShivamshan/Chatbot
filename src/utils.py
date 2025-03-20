# Standard library imports
import os
import re
import logging
import uuid
import json
import pytz
import hashlib
import requests
import textwrap
from typing import List, Dict, Optional, Union
from io import BytesIO
import tempfile
from typing import Any
import subprocess
from collections import defaultdict
from datetime import datetime

# Third-party library imports
import yaml
from PIL import Image
import chromadb
from pydantic import Field, BaseModel
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup, ProcessingInstruction
from rich.console import Console
from rich.panel import Panel

# LangChain imports
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser,JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever,HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Unstructured library imports
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.documents.elements import CompositeElement,Text

# Local imports
from models.Model import Chatbot

# Utility imports
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

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines)) 

def get_parent_id(element):
    """Retrieve the parent ID if available, otherwise use its own element ID."""
    return element["metadata"].get("parent_id", element["element_id"])

# Solution : https://github.com/langchain-ai/langchain/issues/6046 , https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_execute_retrievalqa_chain.py 
class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Optional[Any] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves the relevant documents with a associated scores for each document.

        params
        ------
            - query (str): Query string to search for.

        returns
        -------
            List[Tuple[Document, float]]: A list of tuples, each containing a Document and its corresponding score.
        
        """
        # Perform the similarity search (retrieve more than 5 to ensure top 5 selection)
        docs, scores = zip(*self.vectorstore.similarity_search_with_relevance_scores(query, k=10, score_threshold=0.19))
        
        # Pair documents with their scores
        docs_with_scores = list(zip(docs, scores))
        top_docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)[:5]
    
        for doc, score in top_docs_with_scores:
            doc.metadata["score"] = score
        
        # Return only the top 5 highest-scoring documents
        return [doc for doc, _ in top_docs_with_scores]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

class CustomSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Useful for when you need to answer questions about current or new information through the web."
    web_search_tool: Optional[DuckDuckGoSearchResults] = Field(default=None, exclude=True)  
    wrapper: Optional[DuckDuckGoSearchAPIWrapper] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__()
        # Initialize the wrapper and search tool
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=25, backend="auto")  # Add list of potential websites in 'source' arg if needed
        self.web_search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")

    def _run(self, query: Union[str, list]) -> Union[List[Dict[str, str]], Dict[str, str]]:
        # Check if the query is a list of strings (e.g., multiple search terms)
        if isinstance(query, list):
            results = []
            for single_query in query:
                # Handle each query in the list
                search_results = self.web_search_tool.invoke(single_query)
                # Extract titles, snippets, and links
                results.append([
                    {"title": result.get("title"), "snippet": result.get("snippet"), "url": result.get("link")}
                    for result in search_results
                ])
            return results
        else:
            # If the query is a single string, process it directly
            search_results = self.web_search_tool.invoke(query)
            return [
                {"title": result.get("title"), "snippet": result.get("snippet"), "url": result.get("link")}
                for result in search_results
            ]

class WebscrapperTool(BaseTool):
    name :str = "web_scrape"
    description:str = "Useful for information scrapping for websites"
    llm: Optional[Any] = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True

    def __init__(self,llm:Chatbot):
        super().__init__()
        self.llm = llm

    # Solution : https://github.com/Unstructured-IO/unstructured/issues/3642 
    def parse_html(self,url:str=None) -> List[Text]:
        """
        Use a library like BeautifulSoup to parse HTML from the provided URL and use Unstructured for retreving the elements(chunks).

        params
        ------
            - url (str): The URL of the website to scrape.

        returns
        -------
            List of unstructured.documents.elements.Text 
        """
        try:
            # Add your code here to use a library like BeautifulSoup to scrape HTML from the provided URL.
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()

            # Use BeautifulSoup to parse the HTML
            soup = BeautifulSoup(response.text, "lxml")

            main_content = None

            # Look for common main content containers
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

            # Convert cleaned content to a string
            cleaned_html = str(main_content)

            elements = partition_html(text=cleaned_html)

            return elements
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with web: {e}")
    

    def summarize_text(self, title, text) -> str:
    
        prompt = PromptTemplate(
            template=f"Given the section title: '{title}', summarize the following text:\n{text}",
            input_variables=["text","title"]
        )

        return self.llm.invoke(prompt.format(text=text, title=title))

    def identify_relevant_passages(self, elements: List[Text], keywords: List[str]) -> List[Dict]:
        """
        Identifies and extracts relevant passages from a list of text elements based on given keywords.

        This function processes a list of text elements, identifying relevant sections based on the presence
        of specified keywords. It organizes the relevant content into structured sections, handling different
        types of elements such as titles, narrative text, and list items.

        params
        ------
            - elements (List[Text]): A list of Text objects representing the content to be processed.
            - keywords (List[str]): A list of keywords to identify relevant passages. If None, all passages are considered relevant.

        returns
        -------
            List[Dict]: A list of dictionaries, each representing a relevant section. Each dictionary contains:
                - 'section_id': The ID of the section
                - 'title': The title of the section
                - 'elements': A list of dictionaries, each containing the relevant text and metadata

        The function handles different scenarios:
        - When keywords are provided, it extracts passages containing those keywords.
        - When keywords are None, it extracts all Title and NarrativeText elements.
        - It maintains the structure and hierarchy of the content, including list item positions.
        """
        relevant_sections = {}
        processed_texts = set()
        section_titles = {}
        position_counters = {}

        for element in elements:
            element = element.to_dict()
            element_type = element["type"]

            # Store title if the element is a Title
            if element_type == "Title":
                section_titles[element["element_id"]] = element["text"]

            if keywords is None:
                if element_type in {"Title", "NarrativeText"}:
                    text_key = (element["element_id"], element["text"])

                    if element_type == "NarrativeText":
                        # Ensure we always get the correct parent section ID
                        section_id = element["metadata"].get("parent_id")  # Use parent_id from metadata
                        if not section_id:  
                            continue  

                        if section_id not in relevant_sections:
                            relevant_sections[section_id] = {
                                "section_id": section_id,
                                "title": section_titles.get(section_id, "Unknown Title"),
                                "elements": [],
                            }

                        if text_key not in processed_texts:
                            processed_texts.add(text_key)
                            relevant_sections[section_id]["elements"].append({
                                "text": element["text"],
                            })
            else:
                # Only process relevant element types (Title, NarrativeText, ListItem)
                if element_type in {"Title", "NarrativeText", "ListItem"}:
                    text_lower = element["text"].lower()

                    # Check if any keyword is present in the text
                    if any(keyword in text_lower for keyword in keywords):
                        parent_id = get_parent_id(element)

                        # If it's a NarrativeText, find its corresponding Title's ID
                        if element_type == "NarrativeText":
                            section_id = element["metadata"].get("parent_id")
                            if not section_id:
                                continue  
                        else:
                            section_id = parent_id  # Titles & ListItems use their own ID

                        # Determine the section title
                        section_title = section_titles.get(section_id, "Unknown Title")

                        # Initialize section if not already stored
                        if section_id not in relevant_sections:
                            relevant_sections[section_id] = {
                                "section_id": section_id,
                                "elements": [],
                                "title": section_title,
                            }

                        # Only count positions in sections with ListItems
                        if element_type == "ListItem":
                            if section_id not in position_counters:
                                position_counters[section_id] = 0
                            position_counters[section_id] += 1
                            position = position_counters[section_id]  
                        else:
                            position = None  

                        text_key = (section_id, element["text"])  
                        if text_key not in processed_texts:
                            processed_texts.add(text_key)

                            # Only store the position if it's a ListItem or if the section contains ListItems
                            position_info = {}
                            if element_type == "ListItem":
                                position_info["position"] = position  # Store relative position 

                            relevant_sections[section_id]["elements"].append({
                                "type": element_type,
                                "text": element["text"],
                                "metadata": element["metadata"],
                                **position_info,  # Merge position information 
                            })

        return list(relevant_sections.values())  # Return as a list of dictionaries
    def chunk_large_text(self, title:str, elements: List, max_tokens: int = 1024) -> List[str]:
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=50
        )
        text = " ".join([el["text"] for el in elements])  # Combine section text
        chunks = splitter.split_text(text)  # Split into manageable parts
        return [{"title": title, "text": chunk} for chunk in chunks]
    
    def generate_summaries(self, relevant_sections) -> List[Dict]:
        summaries = []
        for section in relevant_sections:
            title = section["title"]
            elements = section["elements"]
            
            # Chunk the section
            chunks = self.chunk_large_text(title, elements)

            # Summarize each chunk and combine
            section_summary = " ".join(self.summarize_text(chunk["title"], chunk["text"]).content for chunk in chunks)

            summaries.append({"title": title, "summary": section_summary})
        
        return summaries

    def _run(self,keywords: Union[str, List],url:str) -> List[Dict]:

        elements = self.parse_html(url=url)
        keywords = [keywords] if isinstance(keywords, str) else (keywords if isinstance(keywords, list) and keywords else None)
        extracted_data = self.identify_relevant_passages(elements, keywords)
        
        if keywords:
            return extracted_data
        elif keywords is None:
           return self.generate_summaries(extracted_data)
    
    def _arun(self):
        raise NotImplementedError("This tool does not support async")


def format_code(code: str) -> str:
    # return f"```{lang}\n{textwrap.dedent(code)}\n```"
    return f"\n{textwrap.dedent(code)}\n"

class CodeGeneratorTool(BaseTool):
    name: str = "code_tool"
    description: str = "Handles code-related tasks such as syntax highlighting, code analysis, and debugging."
    llm: Optional[Chatbot] = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True

    def __init__(self,llm:Chatbot):
        super().__init__()
        self.llm = llm

    def _run(self, query: str, steps: List[str]) -> dict:
        """
        Uses an LLM to generate Python code based on the query and structured steps.

        Args:
            query (str): The programming task description.
            steps (List[str]): A structured sequence of steps to implement the solution.

        Returns:
            str: The generated Python code.
        """
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        prompt = self._generate_prompt()
        llm_chain = prompt | self.llm | StrOutputParser()
        response = llm_chain.invoke({"query": query, "steps_text": steps_text})
        return response

    def _generate_prompt(self) -> PromptTemplate:
        """
        Formats the query and steps into a structured prompt for the LLM.

        Returns:
            str: A formatted prompt.
        """
        
        try:
            templates = load_ai_template(config_path="config/config.yaml")
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

    def __init__(self, llm: Chatbot):
        super().__init__()
        self.llm = llm

    def _run(self,code,feedbacks:List[Dict]) -> str:
        
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
        try:
            templates = load_ai_template(config_path="config/config.yaml")
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

    def _run(self, code: str) -> dict:
        """
        Analyzes the provided code and returns a structured critique.

        Args:
            code (str): The generated Python code to review.

        Returns:
            dict: A JSON response containing:
                  - critique_feedback (list): Detailed areas of improvement.
                  - is_code_valid (bool): Whether the code is production-ready.
        """
        # Generate the LLM prompt
        prompt = self._generate_review_prompt()
        llm_chain = prompt | self.llm | JsonOutputParser()
        # Call the LLM
        response = llm_chain.invoke({"code":code})
        return response

    def _generate_review_prompt(self) -> str:
        """
        Generates a structured prompt for code review.

        Args:
            code (str): The code snippet to be reviewed.

        Returns:
            str: A structured prompt for the LLM.
        """
        try:
            templates = load_ai_template(config_path="config/config.yaml")
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
    
    def _run(self, query: str) -> str:
        """
        Generates a diagram from a given code snippet.
        It determines whether to create a class diagram or a flowchart.
        """

        # Step 1: Determine the type of diagram (class diagram or flowchart)
        diagram_type_prompt = f"""
        Analyze the following code and determine the best diagram type:
        - If it contains classes and methods, return "classDiagram".
        - If it has function calls, conditionals, or loops, return "flowchart".
        Code:
        {query}
        Return only: "classDiagram" or "flowchart"
        """

        diagram_type = self.llm.invoke(diagram_type_prompt).content

        # Step 2: Generate Mermaid.js diagram
        generate_diagram_prompt = f"""
        Convert the following code into a {diagram_type} in Mermaid.js format.
        Output only valid Mermaid.js syntax.
        Code:
        {query}
        """

        mermaid_diagram = self.llm.invoke(generate_diagram_prompt).content

        return mermaid_diagram

    def _arun(self, query: str):
        """
        This tool does not support asynchronous execution.
        """
        raise NotImplementedError("This tool does not support async")

class DateTimeTool(BaseTool):
    name: str = "date_time_tool"
    description: str = "Handles time-related queries including current time lookup, event schedules, and conversions."
    web_search_tool: Optional[DuckDuckGoSearchResults] = None  
    llm:Optional[Chatbot] = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True

    def __init__(self,llm:Chatbot):
        super().__init__()
        self.llm = llm
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5, backend="api")  
        self.web_search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")

    def _run(self, query: str, category: str):
        """
        Determines which function to use based on category.
        """
        if category == "search_required":
            return self.search_required(query)
        elif category == "event_schedule":
            return self.event_schedule(query)
        elif category == "conversion":
            return self.conversion(query)
        else:
            return "Invalid category."

    def search_required(self, query: str):
        """
        Searches the web for real-time information about time-related events.
        Extracts time-related details from top search results.
        """
        search_results = self.web_search_tool.invoke(query)  # Search the web
        # return search_results
        extracted_info = []

        for result in search_results:
            url = result.get("link")
            if url:
                time_info = self.extract_time_from_url(url)  # Extract time-related details
                if time_info:
                    extracted_info.append({"title": result.get("title"), "url": url, "time_info": time_info})

        return extracted_info if extracted_info else "No relevant time information found."

    def event_schedule(self, query: str):
        """
        Searches the web for event schedules and extracts time information.
        """
        return self.search_required(query)  # Since both need time-related info, reuse search_required.

    def extract_time_from_url(self, url: str) -> str:
        """
        Fetches a webpage and extracts time-related information.
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract time-related text (Look for "7 PM", "18:00", etc.)
            time_keywords = ["AM", "PM", "a.m.", "p.m.", "hour", "time", "schedule"]
            time_info = []

            for tag in soup.find_all(["p", "span", "div"]):  # Scan paragraph and span elements
                text = tag.get_text().strip()
                if any(keyword in text for keyword in time_keywords):
                    time_info.append(text)

            return time_info[:5] if time_info else "No clear time info found."

        except requests.RequestException as e:
            return f"Error accessing {url}: {str(e)}"

    def conversion(self, query: str):
        """
        Uses LLM to handle time zone conversions and date calculations.
        """
        # Provide current time context for the LLM
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"""
        Current system time: {current_time}
        
        I need help with this time or date conversion query: "{query}"
        
        If this involves time zone conversion:
        - Extract the time, source timezone, and target timezone
        - Convert and return only the result in format: "[Time] in [Source TZ] is [Converted Time] in [Target TZ]"
        
        If this involves calculating days between dates:
        - Extract both dates
        - Calculate the number of days between them
        - Return only the result in format: "There are [X] days between [Date1] and [Date2]"
        
        If this involves calculating a relative date:
        - Determine if it's in the past or future
        - Calculate the target date
        - Return only the result in format: "The date [X] [units] [from now/ago] is [Calculated Date]"
        
        Return ONLY the final answer without explanations.
        """
        
        response = self.llm(prompt)
        return response
        
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
    pattern = r"(\/\w+)([^\/]*)"
    
    matches = re.findall(pattern, input_text)
    flag_query_dict = {}
    
    for flag, query in matches:
        flag_query_dict[flag] = query.strip()
    
    return flag_query_dict

def pretty_print_answer(answer):
    """
    Prints the given answer in a formatted panel using the 'rich' library.

    params
    ------
        - answer (str): The answer to be printed.

    returns
    -------
        None. The function prints the answer in a formatted panel.
    """
    console = Console()
    panel = Panel.fit(answer, title="FINAL Answer", border_style="bold cyan")
    console.print(panel)

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



def load_ai_template(config_path: str) -> Dict:
    """
    Load the template present in the config.yaml file

    params
    ------
        - template_name (str): Name of the template to be loaded

    returns
    -------
        Dict: Template configuration loaded from the config.yaml file

    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_file_hash(file:str) -> str:
    """
    Create the hash for a given file  
    
    params
    ------
        - file (str): Path of the given file

    returns
    -------
        str: Hash of the given file
    """
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash


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

    def read_pdf(self, filename: BytesIO):
        """
        Read and partition a PDF file into chunks. It extracts tables, images, and text with specific chunking strategies.

        params
        ------
            filename (BytesIO): A BytesIO object containing the PDF file to be read and partitioned.

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
            if "CompositeElement" in str(type((chunk))):
                texts.append(chunk)

        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
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
    
    @staticmethod
    def get_pdf_title(chunk):
        elements = chunk.metadata.orig_elements
        chunk_title = [el for el in elements if 'Title' in str(type(el))]
        chunk_title[0].to_dict()

        return chunk_title[0].text