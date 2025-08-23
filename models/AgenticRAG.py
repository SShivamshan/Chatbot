from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_core.output_parsers import JsonOutputParser
from models.Model import Chatbot
from src.utils import *
from src.AgentLogger import AgentLogger
from models.WebAgent import WebAgent
from models.PDFAgent import PDFAgent
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# This class is in charge of the AgenticRAG approach in which we can either use external knowledge such as a pdf or search on the web to get the knowledge or both
# Here we don't maintain memory since the own agent takes care of that. 

class GraphState(TypedDict):
    query: Optional[str]
    final_answer: Optional[str]
    pdf_query: Optional[str]
    web_query: Optional[str]
    query_type: Optional[str]
    answer_dict: Optional[Dict]
    filename: Optional[str] 
    sources: Optional[List]
    agent_type: Optional[str] 

class AgenticRAG(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    question_routing: Optional[Any] = Field(default=None, exclude=True)
    logger: Optional[Any] = Field(default=None, exclude=True)
    end_agent:bool = Field(default=True)
    record:bool = Field(default=False)

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        context_length: int = 18000,
        chatbot: Optional[Any] = None,
        log_level: int = logging.INFO,
        pretty_print: bool = True,
        end_agent : bool = True,
        record: bool = False 
    ):
        super().__init__()
        # Initialize LLM
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )

        # Initialize logger
        self.logger = AgentLogger(log_level=log_level, pretty_print=pretty_print,Agent_name="AgenticRAG",record=record)
        self.logger.logger.info(f"Initializing AgenticRAG with model: {self.llm.model}")
        self.end_agent = end_agent
        self.record = record
        
        self.tools = self.initialize_tools()

        QUERY_IDENTIFICATION_PROMPT = self._create_template(template_name="Agentic_RAG_template")
        self.question_routing = QUERY_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()


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
        self.logger.logger.debug("Initializing AgentRAG tools")

        web_agent = WebAgent(base_url=None,model_name=None,chatbot=self.llm,end_agent=False,record=self.record)
        pdf_agent = PDFAgent(base_url=None,model_name=None,chatbot=self.llm,end_agent=False,record=self.record)

        return {"pdf_agent":pdf_agent,"web_agent":web_agent}
    
    def create_graph(self):
        """Create a workflow for Code agent operations."""
        self.logger.logger.info("Creating AgenticRAG workflow graph")
        workflow = StateGraph(GraphState)
        logger = self.logger

        # 1. Identify query type
        def identify_query(state: GraphState) -> GraphState:
            node_name = "identify_query"
            node_start_time = logger.log_node_entry(node_name, state)
            
            result = self.question_routing.invoke({"query": state["query"]})

            updated_state = {
                **state,
                "query_type": result.get("query_type", None),
                "pdf_query": result.get("pdf_query", None),
                "web_query": result.get("web_query", None)
            }
            
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def run_web_agent(state:GraphState) -> GraphState:
            """
            Runs the web agent tool thread
            """
            node_name = "run_web_agent"
            node_start_time = logger.log_node_entry(node_name, state)

            response = self.tools["web_agent"].run(state["query"]) # Contains a dict with keys as final_answer and sourcs(list of dict)

            updated_state = {
                **state,
                "answer_dict": response,
                "agent_type": "web_agent"
            }
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def run_pdf_agent(state:GraphState) -> GraphState:
            """
            Runs the pdf agent tool thread
            """
            node_name = "run_pdf_agent"
            node_start_time = logger.log_node_entry(node_name, state)

            response = self.tools["pdf_agent"].run(state["query"],filename=state["filename"]) # Contains a dict with keys as final_answer and answer_dcit 

            updated_state = {**state,"answer_dict":response,"agent_type":"pdf_agent"}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def run_both_agent(state:GraphState) -> GraphState:
            """
            Runs both PDF and Web agents and combines their outputs
            """
            node_name = "run_both_agent"
            node_start_time = logger.log_node_entry(node_name, state)

            pdf_response = self.tools["pdf_agent"].run(state["pdf_query"],filename=state["filename"])
            web_response = self.tools["web_agent"].run(state["web_query"])

            # Once we ran the web response we will retrieve the sources which will contain the title, url and the content to 
            # be saved inside a temporay vectorstore then use it to retrieve the appropriate info for the web search before 

            sources = web_response.get("sources", [])

            # Build docs for temp vector store (text + image descriptions)
            web_docs = []
            embeddings = OllamaEmbeddings(
                            model="nomic-embed-text:latest",
                            base_url="http://localhost:11434"
                        )
            for src in sources:
                if src.get("content"):
                    web_docs.append(f"Title: {src['title']}\nURL: {src['url']}\nContent: {src['content']}")

            # Add image descriptions
            images = [s for s in sources if "description" in s and s.get("description")]
            for img in images:
                web_docs.append(f"Image: {img['url']}\nDescription: {img['description']}")

            if web_docs:
                temp_store = Chroma.from_texts(web_docs,embeddings)
                relevant_chunks = temp_store.similarity_search(state["web_query"], k=3)

                refined_web_context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            else:
                refined_web_context = web_response.get("final_answer", "")

            # Simple merge logic
            final_answer = (
                f"**PDF Summary:**\n{pdf_response.get('final_answer', '')}\n\n"
                f"**Web Summary (Refined from text & images):**\n{refined_web_context}"
            )
            combined_sources = [pdf_response.get("sources", {})] + sources

            updated_state = {
                **state,
                "answer_dict": {
                    "final_answer": final_answer,
                    "sources": combined_sources
                },
                "agent_type": "both_agents"
            }
            unload_model(logger=self.logger.logger, model_name="nomic-embed-text:latest")

            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def retrieve_values(state:GraphState) -> GraphState:
            node_name = "retrieve_values"
            node_start_time = logger.log_node_entry(node_name, state)
            
            # We now retrieve the values from the state
            response = state["answer_dict"]
            
            updated_state = {**state,"final_answer":response["final_answer"],"sources":response["sources"],"agent_type": state.get("agent_type")}
            logger.log_node_exit(node_name, updated_state, node_start_time)

            return updated_state

        workflow.add_node("identify_query",identify_query)
        workflow.add_node("run_web_agent",run_web_agent)
        workflow.add_node("run_pdf_agent",run_pdf_agent)
        workflow.add_node("run_both_agent",run_both_agent)
        workflow.add_node("retrieve_values",retrieve_values)

        workflow.set_entry_point("identify_query")

        def route_after_identification(state:GraphState) -> List[str]:

            query_type = state.get("query_type") # This gives out a list of strings

            if len(query_type) == 2: # wHICH means both agent needs to be run 
                if state["pdf_query"] != "" and state["web_query"] != "":
                    return ["run_both_agent"]
                else:
                    logging.error(f'ONe of the query is empty, pdf query: {state["pdf_query"]}, web query: {state["web_query"]}')

            else:
                if query_type[0] == "pdf":
                    return["run_pdf_agent"]
                elif query_type[0] == "web":
                    return["run_web_agent"]
        
        workflow.add_conditional_edges("identify_query",route_after_identification)

        workflow.add_edge("run_both_agent","retrieve_values")
        workflow.add_edge("run_pdf_agent","retrieve_values")
        workflow.add_edge("run_web_agent","retrieve_values")
        workflow.set_finish_point("retrieve_values")
        
        return workflow

    def initialize_agent(self):
        """Initialize the agent with LangGraph."""
        self.logger.logger.info("Initializing AgenticRAG workflow")
        # Create the graph
        graph = self.create_graph()
        
        # Compile the graph
        self.logger.logger.info("Compiling AgenticRAG workflow graph")
        executor = graph.compile()
        
        return executor
    
    def run(self, query: str, filename: str = None):
        self.logger.start_agent_run(query)
        self.logger.logger.info(f"Running AgenticRAG with query: {query}")
        executor = self.initialize_agent()
        result = None
        # Initialize state
        state: GraphState = { 
            "query": query,
            "final_answer": None,
            "pdf_query": None,
            "web_query": None,
            "query_type": None,
            "answer_dict": {},
            "filename": filename
        }

        try:
            result = executor.invoke(state)
            self.logger.logger.info("AgenticRAG Execution Completed")
        except Exception as e:
            self.logger.log_error("run", e)
            self.logger.logger.error(f"AgenticRAG run failed: {str(e)}")
            raise e
        finally:
            if self.end_agent:
                self.logger.end_agent_run(result)

        return {
                "final_answer": result.get("final_answer",None),
                "sources": result.get("sources",None),
                "agent": result.get("agent_type", None)
            }