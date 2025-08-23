from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langgraph.graph import StateGraph
from langchain_core.output_parsers import JsonOutputParser
from models.Model import Chatbot
from src.utils import *
from src.AgentLogger import AgentLogger
from models.KBAgent import KBAgent
from models.CodeAgent import CodeAgent
from models.AgenticRAG import AgenticRAG

# Solutions adpated from: https://langchain-ai.github.io/langgraph/concepts/multi_agent/#handoffs-as-tools , 
# https://blog.futuresmart.ai/multi-agent-system-with-langgraph 
class GraphState(TypedDict):
    query: str
    agent_route: Optional[List[str]]
    agent_routes: Optional[List[str]]
    agent_confidence: Optional[float]
    subagent_results: Optional[Dict]
    filename:str 

class SupervisorAgent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    logger: Optional[Any] = Field(default=None, exclude=True)
    agent_routing: Optional[Any] = Field(default=None, exclude=True)
    record:bool = Field(default=False)

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        context_length: int = 18000,
        chatbot: Optional[Any] = None,
        log_level: int = logging.INFO,
        pretty_print: bool = True,
        record: bool = False
    ):
        super().__init__()
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )

        self.logger = AgentLogger(log_level=log_level, pretty_print=pretty_print,record=record)
        self.logger.logger.info(f"Initializing Supervisor Agent with model: {self.llm.model}")
        self.record = record

        self.logger.logger.info("Supervisor LLM initialized")
        self.tools = self.initialize_sub_agents()
        self.logger.logger.info("Sub Agents loaded")

        AGENT_IDENTIFICATION_PROMPT = self._create_template(template_name="Agent_identification_template")
        self.agent_routing = AGENT_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()

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

    def initialize_sub_agents(self):
        self.logger.logger.debug("Initializing sub agents")
        kbagent_tool = KBAgent(chatbot=self.llm,end_agent=False,record=self.record)
        codeagent_tool = CodeAgent(chatbot=self.llm,end_agent=False,record=self.record)
        agenticrag_tool = AgenticRAG(chatbot=self.llm,end_agent = False,record=self.record)
        return {"KBAgent":kbagent_tool,"CodeAgent":codeagent_tool, "AgenticRAG":agenticrag_tool}

    def create_graph(self):
        """Create a workflow for Supervisor agent operations."""
        self.logger.logger.info("Creating SupervisorAgent workflow graph")
        workflow = StateGraph(GraphState)
        logger = self.logger

        def identify_query(state: GraphState) -> GraphState:
            node_name = "supervisor_identify_query"
            node_start_time = logger.log_node_entry(node_name, state)
            result = self.agent_routing.invoke({"query": state["query"]})
            updated_state = {
                **state,
                "agent_route": result.get("agent_route", []),
                "agent_confidence": result.get("agent_confidence", 0)
            }
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def retrieve_agent_results(state: GraphState) -> GraphState:
            agent_routes = state.get("agent_route", [])
            subagent_results = state.get("subagent_results", {})
            
            if not agent_routes:
                logger.logger.warning("No valid agent_route identified.")
                return {**state, "subagent_results": subagent_results}
            
            for route in agent_routes:
                node_name = f"{route}_retriever"
                node_start_time = logger.log_node_entry(node_name, state)

                if route in self.tools:
                    logger.log_tool_call(route, {"query": state.get("query", "")})

                    if route == "AgenticRAG":
                        result = self.tools[route].run(
                            state["query"],
                            filename=state.get("filename")   # only pass if available
                        )
                    else:
                        result = self.tools[route].run(state["query"])
                    
                    subagent_results[route] = result
                else:
                    logger.logger.warning(f"Unknown tool: {route}")

                logger.log_node_exit(node_name, state, node_start_time)

            updated_state = {**state, "subagent_results": subagent_results}
            return updated_state


        def format_agent_results(state: GraphState) -> GraphState:
            logger.logger.debug("Formatting subagent results (stubbed)")
            return state  

        # Add nodes
        workflow.add_node("identify_query", identify_query)
        workflow.add_node("retrieve_agent_results", retrieve_agent_results)
        workflow.add_node("format_agent_results", format_agent_results)

        workflow.set_entry_point("identify_query")
        workflow.add_edge("identify_query", "retrieve_agent_results")
        workflow.add_edge("retrieve_agent_results", "format_agent_results")
        workflow.set_finish_point("format_agent_results")

        return workflow

    def initialize_agent(self):
        self.logger.logger.info("Initializing agent workflow")
        graph = self.create_graph()
        self.logger.logger.info("Compiling agent workflow graph")
        return graph.compile()

    def run(self, query: str,filename: str = None):
        self.logger.start_agent_run(query)
        self.logger.logger.info(f"Running SupervisorAgent...")
        result = None

        state: GraphState = { 
            "query": query,
            "agent_route": None,
            "agent_routes": None,
            "agent_confidence": None,
            "subagent_results": {},
            "filename": filename
        }

        try:
            executor = self.initialize_agent()
            result = executor.invoke(state)
            self.logger.logger.info(f"SupervisorAgent finished running...")
        except Exception as e:
            self.logger.log_error("agent_run", e)
            self.logger.logger.error(f"SupervisorAgent run failed: {str(e)}")
            return f"Error running SupervisorAgent: {str(e)}"
        finally:
            self.logger.end_agent_run(result,False)
            return result["subagent_results"]
    