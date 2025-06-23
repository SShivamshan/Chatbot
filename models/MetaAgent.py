from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langgraph.graph import StateGraph
from langchain_core.output_parsers import JsonOutputParser
from models.Model import Chatbot
from src.utils import *
from src.AgentLogger import AgentLogger
from models.WebAgent import WebAgent

class GraphState(TypedDict):
    query: str
    agent_route: Optional[List[str]]
    agent_confidence: Optional[float]

class MetaAgent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    ai_template: dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)

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
        self.logger = AgentLogger(log_level=log_level, pretty_print=pretty_print)
        self.logger.logger.info(f"Initializing Agent with model: {model_name}")

        # Initialize LLM
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )

        self.logger.logger.info("LLM initialized")
        self.tools = self.initialize_agents()
        self.ai_template = load_ai_template('config/config.yaml')
        self.logger.logger.info("Tools and templates loaded")

        AGENT_IDENTIFICATION_PROMPT = self._create_template(template_name="Agent_identification_template")
        self.agent_routing = AGENT_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()


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
        self.logger.logger.debug("Initializing tools")
        # Instantiate the tools
        webagent_tool = WebAgent(chatbot=self.llm)
        tools = {
            "web_search": webagent_tool,
        }
        return tools
    
    def create_graph(self):
        workflow = StateGraph(GraphState)

        def identify_query(state: GraphState) -> GraphState:
            pass 

        def retrieve_agent_results(state: GraphState) -> GraphState:
            """
            Retrieves the agents outputs
            """

        def format_agent_results(state:GraphState) ->GraphState:
            """
            Formats agent results based on the type of task and type of printing for the app
            """
    
    def initialize_agent(self):
        """Initialize the agent with LangGraph."""
        self.logger.logger.info("Initializing agent workflow")
        # Create the graph
        graph = self.create_graph()
        
        # Compile the graph
        self.logger.logger.info("Compiling agent workflow graph")
        executor = graph.compile()
        
        return executor
    
    def run(self, query: str):
        """Run the agent with the given query."""
        self.logger.start_agent_run(query)
        self.logger.logger.info(f"Running agent with query: {query}")
        
        try:
            executor = self.initialize_agent()
            result = executor.invoke({"query": query})
            # Store the results in a dictionary
            final_result = {
                "query": result.get("query"),
                "final_answer": result.get("final_answer"),
                "sources": result.get("sources")  
            }

            self.logger.end_agent_run(result)
            return {
                "answer": final_result.get("final_answer", "No answer generated."),
                "sources": final_result.get("sources", [])
            }
        except Exception as e:
            self.logger.log_error("agent_run", e)
            self.logger.logger.error(f"Agent run failed: {str(e)}")
            return f"Error running agent: {str(e)}"