from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain.storage import InMemoryStore
from models.Model import Chatbot
from threading import Thread
from queue import Queue, Empty
from typing import Callable
from src.utils import *
import sqlite3
import tempfile
import shutil
from src.AgentLogger import AgentLogger
from models.RAG import RAG

## PDF agent is an agent itself to go through vector databases(PDF) for releveant information
# Finish retrieve vlaues nodes, problem: the state seems to disappear embedding dimension for the db 
# No need for memory since we have RAG that does keep in memory

class GraphState(TypedDict):
    query: str
    final_answer: Optional[str]
    online: bool
    sources: Optional[List]
    pdf_url: Optional[str]
    pdf_category: str
    vector_store_populated: bool

class PDFAgent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    logger: Optional[Any] = Field(default=None, exclude=True)
    vectorstore: Optional[Any] = Field(default=None, exclude=True)
    pdf_type_: Optional[Any] = Field(default=None, exclude=True)
    uploaded_filename:str = Field(default="")
    offline_saved: bool = Field(default=False)
    online_saved: bool = Field(default=False)
    # custom_retriever: Optional[CustomRetriever] = Field(default=None, exclude=True)
    docstore: Optional[Any] = Field(default=None, exclude=True)
    # rag: Optional[Any] = Field(default=None,exclude=True)
    temp_dir: Optional[Any] = Field(default=None, exclude=True)
    last_offline_file: str = Field(default=None, exclude=True)
    last_online_url: str = Field(default=None, exclude=True)

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        context_length: int = 18000,
        chatbot: Optional[Any] = None,
        log_level: int = logging.INFO,
        pretty_print: bool = True
    ):
        """Initialize the agent with modern LangChain patterns."""
        super().__init__()
        # Initialize logger
        self.logger = AgentLogger(log_level=log_level, pretty_print=pretty_print,Agent_name="PDF Agent")
        self.logger.logger.info(f"Initializing CodeAgent with model: {model_name}")

        # Initialize LLM
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )
        self.vectorstore = Vectordb(NAME="PDFAGENT_knowledge_base")
        self.docstore = InMemoryStore()
        # State of the pdf if they are saved or not(means are they set onto the vector and doc stores)
        self.uploaded_filename = ""
        self.online_saved = False
        self.offline_saved = False
        self.last_online_url = None
        self.last_offline_file = None
        self.temp_dir = tempfile.mkdtemp()
        # self.saved_paths = {}
        # self.saved_tables = {}
        self.logger.logger.info("Code Agent LLM initialized")

        # Initialize components
        self.tools = self.initialize_tools()
        PDF_TYPE_IDENTIFICATION_PROMPT = self._create_template(template_name="PDF_Agent_template")
        self.pdf_type_ = PDF_TYPE_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()

        self.logger.logger.info("PDF Agent Tools and templates loaded")
    
    def __del__(self):
        try:
            if hasattr(self, 'temp_dir') and isinstance(self.temp_dir, (str, bytes, os.PathLike)):
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
        except Exception as e:
            pass  # Avoid throwing in destructor
    
    def _save_images(self, img_base64_list: List[str], filename: str, img_uids: List[float]) -> List[str]:
        """
        Temporarily saves base64 images to a temp directory and returns file paths
        """
        try:
            file_paths = []
            for i, image_b64 in enumerate(img_base64_list):
                file_path = os.path.join(self.temp_dir, f'{filename}_image_{i}.jpeg')

                # Convert base64 to image
                img_bytes = self.tools["pdf_reader"].base64_to_image(image_b64)
                img = self.tools["pdf_reader"].create_image_from_bytes(img_bytes)

                img.save(file_path, optimize=True, quality=95)
                # self.saved_paths[str(img_uids[i])] = file_path
                file_paths.append(file_path)
            self.logger.logger.info("PDF images are temporarily saved")
            return file_paths
        
        except Exception as e:
            self.logger.logger.error(f"An error occurred while saving images: {e}")
            raise Exception(f"An error occurred while saving images: {e}")
        
    def _create_template(self,template_name:str) -> PromptTemplate:
        try:
            self.logger.logger.debug("Loading template")
            templates = load_ai_template(config_path="config/config.yaml")
            template_config = templates["Agent_templates"][template_name]
            template = template_config["template"]
            input_variables = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        except Exception as e:
            self.logger.log_error(f"_create_template({template_name})", e)
            raise ValueError(f"Failed to initialize template: {template_name}")
        
    def initialize_tools(self):
        self.logger.logger.debug("Initializing PDF Agent tools")
        pdf_reader = PDF_Reader()
        custom_retriever = CustomRetriever(vectorstore=self.vectorstore.vector_store,docstore=self.docstore)
        rag = RAG(multi_retriever=custom_retriever, chatbot=self.llm , chat_data_manager=None)

        return {'pdf_reader':pdf_reader,'rag':rag}
    
    def run_through_thread(self, func:Callable, *args, **kwargs):  
        """
        Start a background thread to run a funtion
        """
        result_queue = Queue()

        def thread_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(None)
        save_thread = Thread(target=thread_wrapper, args=args, kwargs=kwargs, daemon=True)
        save_thread.start()

        # Wait for the thread to complete and retrieve the result
        save_thread.join()
        try:
            return result_queue.get_nowait()  # Get result without blocking
        except Empty:
            self.logger.error("Queue is empty; no result returned from thread.")
            return None
    
    def populate_vector_store(self,file:str,mode:str=None):
        table_summaries = []
        image_summaries = []
        id_key = "doc_id"
        chunks = []
        if mode == "offline":
            chunks = self.tools["pdf_reader"].read_pdf(file)
            self.online_saved = True
        if mode == "online":
            chunks = self.tools["pdf_reader"].read_pdf_online(file)
            self.offline_saved = True
            
        filename = chunks[0].metadata.filename
        texts,tables,images_64_list = self.tools["pdf_reader"].separate_elements(chunks)
        chunked_texts, _ = split_chuncks(texts,filename=filename)
        
        if tables:
            table_summaries = self.tools["pdf_reader"]._get_summaries_table(tables = tables,chatbot=self.llm)
            
        if images_64_list:
            llm_image = Chatbot(model="llava:7b")
            image_summaries = self.tools["pdf_reader"]._get_summaries_image(images=images_64_list,chatbot=llm_image)
            llm_image.unload_model()
            del llm_image

        if chunked_texts:
            self.vectorstore.populate_vector(documents=chunked_texts)

        if image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in images_64_list]
            # First we save the image temporarily
            file_paths = self.run_through_thread(self._save_images,img_base64_list=images_64_list,filename = filename,img_uids= img_ids)
            summary_docs = [
                Document(page_content=s, metadata={id_key: img_ids[i], "filename":filename})
                        for i, s in enumerate(image_summaries)
            ]
            self.vectorstore.populate_vector(documents=summary_docs)
            self.docstore.mset(list(zip(img_ids, file_paths)))

        if table_summaries:
            tab_ids = [str(uuid.uuid4()) for _ in tables]
            table_docs = [
                Document(page_content=s, metadata={id_key: tab_ids[i], "filename":filename})
                    for i, s in enumerate(table_summaries)
            ]
            self.vectorstore.populate_vector(documents=table_docs)
            self.docstore.mset(list(zip(tab_ids,tables)))

        self.logger.logger.info("Embeddings were unloaded from the GPU")
        unload_model(logger=self.logger.logger, model_name="nomic-embed-text:latest")

        self.logger.logger.info(f"Created the vectorstore for {mode} pdf")
        
    def create_graph(self):
        """Create a workflow for Code agent operations."""
        self.logger.logger.info("Creating PDF agent workflow graph")
        workflow = StateGraph(GraphState)
        logger = self.logger 

        # 1. Identify pdf format type online or offline pdf or both
        def identify_pdf_format(state: GraphState) -> GraphState:
            node_name = "identify_query"
            node_start_time = logger.log_node_entry(node_name, state)

            # Check if identification has already been done AND files/URLs are unchanged
            if "pdf_category" in state and state["pdf_category"] in ["online", "offline", "both"]:
                if state["pdf_category"] == "online":
                    if self.last_online_url == state.get("pdf_url", None):
                        logger.log_node_exit(node_name, state, node_start_time)
                        return state
                elif state["pdf_category"] == "offline":
                    if self.last_offline_file == self.uploaded_filename:
                        logger.log_node_exit(node_name, state, node_start_time)
                        return state
                elif state["pdf_category"] == "both":
                    if (self.last_online_url == state.get("pdf_url", None) and 
                        self.last_offline_file == self.uploaded_filename):
                        logger.log_node_exit(node_name, state, node_start_time)
                        return state

            # Otherwise, run identification again
            result = self.pdf_type_.invoke({
                "query": state["query"],
                "online": state["online"]
            })

            updated_state = {
                **state,
                "pdf_category": result.get("pdf_category", []),
                "pdf_url": result.get("url", None)
            }

            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def populate_vector_store_node(state: GraphState) -> GraphState:
            node_name = "populate_vector_store"
            node_start_time = self.logger.log_node_entry(node_name, state)

            mode = state.get("pdf_category")  # "online", "offline", or "both"
            url = state.get("pdf_url", None)
            updated = False

            if mode in ["online", "both"]:
                if self.last_online_url != url:
                    self.populate_vector_store(file=url, mode="online")
                    self.last_online_url = url
                    self.online_saved = True
                    updated = True

            if mode in ["offline", "both"]:
                if self.last_offline_file != self.uploaded_filename:
                    self.populate_vector_store(file=self.uploaded_filename, mode="offline")
                    self.last_offline_file = self.uploaded_filename
                    self.offline_saved = True
                    updated = True

            updated_state = {
                **state,
                "vector_store_populated": updated or self.online_saved or self.offline_saved,
            }
            self.logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def run_online_tool(state: GraphState) -> GraphState:
            """
            Runs the online tool thread
            """
            node_name = "run_online_tool"
            node_start_time = logger.log_node_entry(node_name, state)
            response = None

            if self.online_saved:
                response = self.tools["rag"].run(state["query"])

            updated_state = {**state,"final_answer":response}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def run_offline_tool(state: GraphState) -> GraphState:
            """
            Runs the offline tool thread
            """
            node_name = "run_offline_tool"
            node_start_time = logger.log_node_entry(node_name, state)

            response = None

            if self.offline_saved:
                response = self.tools["rag"].run(state["query"])

            updated_state = {**state,"final_answer":response}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def run_both_tool(state: GraphState) -> GraphState:
            """
            Runs both tools 
            """
            node_name = "run_both_tool"
            node_start_time = logger.log_node_entry(node_name, state)

        def retrieve_values(state:GraphState) -> GraphState:
            node_name = "retrieve_values"
            node_start_time = logger.log_node_entry(node_name, state)
            

        def generate_answer(state: GraphState) -> GraphState:
            pass 

        workflow.add_node("identify_pdf_format",identify_pdf_format)
        workflow.add_node("populate_vector_store", populate_vector_store_node)
        workflow.add_node("run_online_tool",run_online_tool)
        workflow.add_node("run_offline_tool",run_offline_tool)
        workflow.add_node("run_both_tool",run_both_tool)
        workflow.add_node("retrieve_values",retrieve_values)
        workflow.add_node("generate_answer",generate_answer)

        workflow.set_entry_point("identify_pdf_format")
        
        def route_after_identification(state: GraphState) -> List[str]:
            return ["populate_vector_store"]
        
        def route_after_vector_store(state: GraphState) -> List[str]:
            pdf_category = state.get("pdf_category")
            if pdf_category == "online":
                return ["run_online_tool"]
            elif pdf_category == "offline":
                return ["run_offline_tool"]
            else:
                return ["run_both_tool"]

        workflow.add_conditional_edges("identify_pdf_format",route_after_identification)
        workflow.add_conditional_edges("populate_vector_store", route_after_vector_store)
        
        workflow.add_edge("run_both_tool","retrieve_values")
        workflow.add_edge("run_offline_tool","retrieve_values")
        workflow.add_edge("run_online_tool","retrieve_values")
        workflow.add_edge("retrieve_values","generate_answer")
        workflow.set_finish_point("generate_answer")

        return workflow
    
    def initialize_agent(self):
        """Initialize the agent with LangGraph."""
        self.logger.logger.info("Initializing PDFAgent workflow")
        # Create the graph
        graph = self.create_graph()
        
        # Compile the graph
        self.logger.logger.info("Compiling PDFAgent workflow graph")
        executor = graph.compile()
        
        return executor
    
    def run(self, query: str,filename:str=None):
        """Run the agent with the given query."""
        self.logger.start_agent_run(query)
        self.logger.logger.info(f"Running PDFAgent with query: {query}")
        
        try:
            executor = self.initialize_agent()
            uploaded = False if filename is None else True
            self.uploaded_filename = filename
            result = executor.invoke({"query": query,"online":uploaded})

            self.logger.end_agent_run(result)
            return result["final_answer"]
        except Exception as e:
            self.logger.log_error("agent_run", e)
            self.logger.logger.error(f"PDFAgent run failed: {str(e)}")
            return f"Error running PDFAgent: {str(e)}"
    
