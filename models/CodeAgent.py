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
    action_type: Optional[str]
    core_steps: Optional[List[str]]
    steps: Optional[List[str]]
    generated_code: Optional[str]
    diagram_image: Optional[str]
    critique_feedback: Optional[str]
    is_code_valid: Optional[bool]
    iterations_left: Optional[int]

class CodeAgent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    code_query_identification: Optional[Any] = Field(default=None, exclude=True)
    code_list: Optional[Any] = Field(default=None, exclude=True)
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
        """Initialize the agent with modern LangChain patterns."""
        super().__init__()
        # Initialize logger
        self.logger = AgentLogger(log_level=log_level, pretty_print=pretty_print,Agent_name="Code Agent")
        self.logger.logger.info(f"Initializing CodeAgent with model: {model_name}")

        # Initialize LLM
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )
        self.logger.logger.info("Code Agent LLM initialized")

        # Initialize components
        self.tools = self.initialize_tools()
        CODE_QUERY_IDENTIFICATION_PROMPT = self._create_template(template_name = "Code_query_identification_template")
        self.code_query_identification = CODE_QUERY_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()

        CODE_LIST_PROMPT = self._create_template(template_name="Code_generation_list_template")
        self.code_list = CODE_LIST_PROMPT | self.llm | JsonOutputParser()

        self.logger.logger.info("Code Agent Tools and templates loaded")

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
        self.logger.logger.debug("Initializing Code Agent tools")
        # Instantiate the tools
        generate_code_tool = CodeGeneratorTool(llm=self.llm)
        code_review_tool = CodeReviewTool(llm=self.llm)
        code_corrector_tool = CodeCorrectorTool(llm=self.llm)
        diagram_creator_tool = CodeDiagramCreator(llm=self.llm)

        tools = {
            "code_generator": generate_code_tool,
            "code_critique": code_review_tool,
            "code_corrector": code_corrector_tool,
            "diagram_creator": diagram_creator_tool
        }
        return tools
    
    def create_graph(self):
        """Create a workflow for Code agent operations."""
        self.logger.logger.info("Creating Code agent workflow graph")
        workflow = StateGraph(GraphState)
        logger = self.logger

        # 1. Code query identification
        def code_query_identification(state: GraphState) -> GraphState:
            node_name = "code_query_identification"
            node_start_time = logger.log_node_entry(node_name, state)
            
            logger.logger.info(f"Performing code query identification on: {state['query']}")
            result = self.code_query_identification.invoke({"query": state["query"]})
            updated_state = {**state, "action_type": result.get("action_type", "")}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def code_research(state: GraphState) -> GraphState:
            node_name = "code_research"
            node_start_time = logger.log_node_entry(node_name, state)
            
            logger.logger.info(f"Performing code research on: {state['query']}")
            result = self.code_list.invoke({"query": state["query"]})
            
            updated_state = {**state, "steps": result.get("steps", [])}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def critique_code(state: GraphState) -> GraphState:
            node_name = "critique_code"
            node_start_time = self.logger.log_node_entry(node_name, state)

            if not state.get("generated_code"):
                self.logger.logger.warning("No generated code to critique.")
                return state  # Skip if no code

            # Call the critique tool (runs tests & linting)
            critique_result = self.tools["code_critique"]._run(state["generated_code"])
            # Update the steps part
            state["core_steps"] = state["steps"]
            state["steps"] = critique_result["feedback"]
            
            updated_state = {
                **state,
                "critique_feedback": critique_result["feedback"],
                "is_code_valid": critique_result["is_valid"],
                "iterations_left": state.get("iterations_left", 3) - 1
            }
            self.logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def correct_code(state:GraphState) -> GraphState:
            node_name = "correct_code"
            node_start_time = self.logger.log_node_entry(node_name, state)

            if not state.get("generated_code"):
                self.logger.logger.warning("No generated code to correct.")
                return state  # Skip if no code

            # Call the code correction tool (runs automated corrections)
            corrected_code = self.tools["code_corrector"]._run(state["generated_code"],state["steps"])
            updated_state = {**state, "generated_code": corrected_code}

            self.logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def generate_code(state: GraphState) -> GraphState:
            node_name = "generate_code"
            node_start_time = self.logger.log_node_entry(node_name, state)

            # Call the code generation tool
            code_result = self.tools["code_generator"]._run(query=state["query"],steps=state.get("steps", []))

            updated_state = {**state, "generated_code": code_result}
            self.logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def create_diagram(state:GraphState) -> GraphState:
            node_name = "create_diagram"
            node_start_time = self.logger.log_node_entry(node_name, state)

            if not state.get("generated_code"):
                self.logger.logger.warning("No generated code to create a diagram.")
                return state  # Skip if no code

            # Call the diagram creation tool (runs diagram generation)
            diagram_result = self.tools["diagram_creator"]._run(state["query"])
            updated_state = {**state, "diagram_image": diagram_result["mermaid"]}

            self.logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def prepare_code_for_critique(state: GraphState) -> GraphState:
            """Updates the state to move the provided code (query) into generated_code."""
            node_name = "prepare_code_for_critique"
            node_start_time = self.logger.log_node_entry(node_name, state)

            updated_state = {**state, "generated_code": state["query"]} 

            self.logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def code_done(state: GraphState) -> GraphState:
            node_name = "code_done"
            node_start_time = self.logger.log_node_entry(node_name, state)

            updated_state = {**state, "final_answer":state["generated_code"]}
            self.logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
            
        workflow.add_node("code_research", code_research)
        workflow.add_node("generate_code", generate_code)
        workflow.add_node("critique_code", critique_code)
        workflow.add_node("code_query_identification", code_query_identification)
        workflow.add_node("prepare_code_for_critique", prepare_code_for_critique)
        workflow.add_node("correct_code", correct_code)
        workflow.add_node("create_diagram", create_diagram)
        workflow.add_node("code_done",code_done)

        workflow.set_entry_point("code_query_identification")

        def route_code_query_identification(state: GraphState) -> List[str]:
            action_type = state.get("action_type")
            if action_type == "generate":
                return ["generate_code"]
            elif action_type in ["correct", "critique"]: # TO correct the code we first go through the review process so that it could allow us to correct it based on these feedbacks
                return ["prepare_code_for_critique"]
            elif action_type == "diagram":
                return ["create_diagram"]
            
        def route_code_loop(state: GraphState) -> List[str]:
            """Loop between generation and critique until the code is valid or iterations run out."""
            action_type = state.get("action_type")
            if state.get("is_code_valid") or state.get("iterations_left", 3) <= 0 or action_type in ["diagram"] :
                return ["code_done"]
            return ["correct_code"]

        workflow.add_conditional_edges("critique_code", route_code_loop)
        workflow.add_conditional_edges("code_query_identification", route_code_query_identification) 

        # Add edges
        workflow.add_edge("code_research","generate_code")
        workflow.add_edge("generate_code", "critique_code")
        workflow.add_edge("correct_code", "critique_code")
        workflow.add_edge("prepare_code_for_critique", "critique_code")
        workflow.add_edge("create_diagram", "code_done")

        workflow.set_finish_point("code_done")

        return workflow
    
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

            self.logger.end_agent_run(result)
            return {
                "generated_code": result.get("generated_code"),
                "core_steps": result.get("core_steps",""),
                "critique_feedback": result.get("critique_feedback"),
                "diagram_image": result.get("diagram_image","")
            }
            
        except Exception as e:
            self.logger.log_error("agent_run", e)
            self.logger.logger.error(f"Agent run failed: {str(e)}")
            return f"Error running agent: {str(e)}"
