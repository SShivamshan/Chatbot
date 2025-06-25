from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
from models.Model import Chatbot
from src.utils import *
from src.AgentLogger import AgentLogger


class GraphState(TypedDict):
    query: str
    final_answer: Optional[str]
    sources: Optional[List]
    domain: str
    subdomain: str

class KBAgent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    knowledge_result: Optional[Any] = Field(default=None, exclude=True)
    domain_context: Optional[Any] = Field(default=None, exclude=True)
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
        self.logger = AgentLogger(log_level=log_level, pretty_print=pretty_print,Agent_name="Knowledge Base Agent")
        self.logger.logger.info(f"Initializing Knowledge Base agent with model: {model_name}")

        # Initialize LLM
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )
        self.logger.logger.info("Web Agent LLM initialized")
        
        DOMAIN_CONTEXT_PROMPT = self._create_template(template_name="Domain_context_template")
        self.domain_context = DOMAIN_CONTEXT_PROMPT | self.llm | JsonOutputParser()

        KNOWLEDGE_AGENT_PROMPT = self._create_template(template_name="Knowledge_base_template")
        self.knowledge_result = KNOWLEDGE_AGENT_PROMPT | self.llm | JsonOutputParser()

        self.logger.logger.info("Knowledge base agent template loaded")

    def _create_template(self, template_name: str) -> PromptTemplate:
        try:
            self.logger.logger.debug("Loading template")
            templates = load_ai_template(config_path="config/config.yaml")
            template_config = templates["Agent_templates"][template_name]
            template = template_config["template"]
            input_vars = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(template=template, input_variables=input_vars)
        except Exception as e:
            self.logger.log_error(f"_create_template({template_name})", e)
            raise ValueError(f"Failed to initialize template: {template_name}")
    
    def create_graph(self):
        """Create a LangGraph workflow for agent operations."""
        self.logger.logger.info("Creating Knowledge Base Agent workflow graph")
        workflow = StateGraph(GraphState)
        logger = self.logger

        def identify_domain(state: GraphState) -> GraphState:
            node_name = "identify_domain"
            node_start_time = logger.log_node_entry(node_name, state)
            
            result = self.domain_context.invoke({"query": state["query"]})
            updated_state = {
                **state,
                "domain": result.get("domain",""),
                "subdomain": result.get("subdomain","")
            }

            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def retrieve_knowledge(state: GraphState) -> GraphState:
            node_name = "retrieve_knowledge"
            node_start_time = logger.log_node_entry(node_name, state)

            result = self.knowledge_result.invoke({"query": state["query"],"domain":state["domain"],"subdomain":state["subdomain"]})
            updated_state = {
                **state,
                "final_answer":result.get("answer","")
            }
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        # Add nodes
        workflow.add_node("identify_domain", identify_domain)
        workflow.add_node("retrieve_knowledge", retrieve_knowledge)

        workflow.set_entry_point("identify_domain") # Start point
        
        workflow.add_edge("identify_domain", "retrieve_knowledge")
        workflow.set_finish_point("retrieve_knowledge")
        
        return workflow
    
    def initialize_agent(self):
        """Initialize the agent with LangGraph."""
        self.logger.logger.info("Initializing Knowledge base agent workflow")
        # Create the graph
        graph = self.create_graph()
        
        # Compile the graph
        self.logger.logger.info("Compiling Knowledge base agent workflow graph")
        executor = graph.compile()
        
        return executor
    
    def run(self, query: str):
        """Run the agent with the given query."""
        self.logger.start_agent_run(query)
        self.logger.logger.info(f"Running Knowledge base agent with query: {query}")
        
        try:
            executor = self.initialize_agent()
            result = executor.invoke({"query": query})

            self.logger.end_agent_run(result)
            return {
                "answer": result.get("final_answer", "No answer generated.").content if isinstance(result.get("final_answer", "No answer generated."), AIMessage) else result.get("final_answer", "No answer generated.")
            }
        except Exception as e:
            self.logger.log_error("agent_run", e)
            self.logger.logger.error(f"Knowledge base Agent run failed: {str(e)}")
            return f"Error running Knowledge base agent: {str(e)}"