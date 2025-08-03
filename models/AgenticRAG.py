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

# This class is in charge of the AgenticRAG approach in which we can either use external knowledge such as a pdf or search on the web to get the knowledge or both
# Here we don't maintain memory since the own agent takes care of that. 

## TODO
# Web agent works with the agentic RAG but not the pdf agent at the populate_vectorstore node

class GraphState(TypedDict):
    query: str
    final_answer: Optional[str]
    pdf_query: Optional[str]
    web_query: Optional[str]
    query_type: Optional[str]
    answer_dict: Dict
    filename: str 

class AgenticRAG(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    question_routing: Optional[Any] = Field(default=None, exclude=True)
    logger: Optional[Any] = Field(default=None, exclude=True)

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        context_length: int = 18000,
        chatbot: Optional[Any] = None,
        log_level: int = logging.INFO,
        pretty_print: bool = True
    ):
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

        web_agent = WebAgent(base_url=None,model_name=None,chatbot=self.llm,end_agent=False)
        pdf_agent = PDFAgent(base_url=None,model_name=None,chatbot=self.llm,end_agent=False)

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

            updated_state = {**state,"answer_dict":response}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def run_pdf_agent(state:GraphState) -> GraphState:
            """
            Runs the pdf agent tool thread
            """
            node_name = "run_pdf_agent"
            node_start_time = logger.log_node_entry(node_name, state)

            response = self.tools["pdf_agent"].run(state["query"]) # Contains a dict with keys as final_answer and answer_dcit 

            updated_state = {**state,"answer_dict":response}
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

            # Simple merge logic
            final_answer = f"PDF: {pdf_response.get('final_answer', '')}\n\nWeb: {web_response.get('final_answer', '')}"
            combined_sources = [pdf_response.get("answer_dict", {})] + web_response.get("sources", [])

            updated_state = {
                **state,
                "answer_dict": {
                    "final_answer": final_answer,
                    "sources": combined_sources
                }
            }

            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def retrieve_values(state:GraphState) -> GraphState:
            node_name = "retrieve_values"
            node_start_time = logger.log_node_entry(node_name, state)
            
            # We now retrieve the values from the state
            response = state["answer_dict"]
            
            updated_state = {**state,"final_answer":response["final_answer"]}
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
            self.logger.end_agent_run(result)

        return {
                "answer_dict": result.get("answer_dict",None)
            }

        

        
