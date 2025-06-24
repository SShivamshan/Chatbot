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
    subagent_results: Optional[List]

class SupervisorAgent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    ai_template: dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    logger: Optional[Any] = Field(default=None, exclude=True)
    agent_routing: Optional[Any] = Field(default=None, exclude=True)

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
        self.logger = AgentLogger(log_level=log_level, pretty_print=pretty_print)
        self.logger.logger.info(f"Initializing Supervisor Agent with model: {model_name}")

        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )

        self.logger.logger.info("Supervisor LLM initialized")
        self.tools = self.initialize_sub_agents()
        self.ai_template = load_ai_template('config/config.yaml')
        self.logger.logger.info("Sub Agents loaded")

        AGENT_IDENTIFICATION_PROMPT = self._create_template(template_name="Agent_identification_template")
        self.agent_routing = AGENT_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()

    def _create_template(self, template_name: str) -> PromptTemplate:
        try:
            self.logger.logger.debug("Loading template")
            template_config = self.ai_template["Agent_templates"][template_name]
            template = template_config["template"]
            input_vars = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(template=template, input_variables=input_vars)
        except Exception as e:
            self.logger.log_error(f"_create_template({template_name})", e)
            raise ValueError(f"Failed to initialize template: {template_name}")

    def initialize_sub_agents(self):
        self.logger.logger.debug("Initializing sub agents")
        webagent_tool = WebAgent(chatbot=self.llm)
        return {"WebAgent": webagent_tool}

    def create_graph(self):
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
            node_name = f"{'_'.join(state.get('agent_route', ['unknown']))}_retriever"
            node_start_time = logger.log_node_entry(node_name, state)

            if state.get("agent_route"):
                results = []
                for route in state["agent_route"]:
                    if route in self.tools:
                        logger.log_tool_call(route, {"query": state.get("query", "")})
                        result = self.tools[route].run(state["query"])
                        results.append(result)
                    else:
                        logger.logger.warning(f"Unknown tool: {route}")
                updated_state = {**state, "subagent_results": results}
            else:
                logger.logger.warning("No valid agent_route identified.")
                updated_state = {**state, "subagent_results": []}

            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def format_agent_results(state: GraphState) -> GraphState:
            logger.logger.debug("Formatting subagent results (stubbed)")
            return state  # Implement formatting later

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

    def run(self, query: str):
        self.logger.start_agent_run(query)
        self.logger.logger.info(f"Running SupervisorAgent...")
        try:
            executor = self.initialize_agent()
            result = executor.invoke({"query": query})
            self.logger.end_agent_run(result)
            return result  # ✅ Return the result to user
        except Exception as e:
            self.logger.log_error("agent_run", e)
            self.logger.logger.error(f"Agent run failed: {str(e)}")
            return f"Error running agent: {str(e)}"
    