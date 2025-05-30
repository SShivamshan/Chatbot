import yaml
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.tools import Tool
from langgraph.graph import StateGraph
from langchain.schema import BaseRetriever,AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools.render import render_text_description_and_args
from models.Model import Chatbot
from src.utils import *
from src.AgentLogger import AgentLogger

# SOLUTION adopted from : https://medium.com/@sahin.samia/how-to-build-a-interactive-personal-ai-research-agent-with-llama-3-2-b2a390eed63e 
# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot 
# https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/ 
class GraphState(TypedDict):
    query: str
    category: Optional[List[str]]
    confidence: Optional[float]
    reasoning: Optional[str]
    required_tools: Optional[List[str]]
    keywords: Optional[List[str]]
    search_results: Optional[List[Dict[str, Any]]]
    scrape_results: Optional[List[Dict[str, Any]]]
    scrape_url: Optional[str]
    elements_to_retrieve: Optional[List[str]]
    final_answer: Optional[str]
    final_json: Optional[Dict[str, Any]]
    action_type: Optional[str]
    steps: Optional[List[str]]
    generated_code: Optional[str]
    diagram_image: Optional[str]
    critique_feedback: Optional[str]
    is_code_valid: Optional[bool]
    iterations_left: Optional[int]

class Agent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    ai_template: dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    question_routing: Optional[Any] = Field(default=None, exclude=True)
    question_to_keywords: Optional[Any] = Field(default=None, exclude=True)
    query_web_scrape: Optional[Any] = Field(default=None, exclude=True)
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
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            context_length=context_length
        )
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
        # Initialize components
        self.tools = self.initialize_tools()
        self.ai_template = load_ai_template('config/config.yaml')
        self.logger.logger.info("Tools and templates loaded")
        # Initialize agent chains
        QUERY_IDENTIFICATION_PROMPT = self._create_template(template_name="Query_identification_template")
        self.question_routing = QUERY_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()
        
        QUERY_WEB_PROMPT = self._create_template(template_name="Query_web_template")
        self.question_to_keywords = QUERY_WEB_PROMPT | self.llm | JsonOutputParser()

        QUERY_WEB_SCRAPE_CLASSIFICATION_PROMPT = self._create_template(template_name="Query_web_scrape_classification")
        self.query_web_scrape = QUERY_WEB_SCRAPE_CLASSIFICATION_PROMPT | self.llm | JsonOutputParser()

        CODE_QUERY_IDENTIFICATION_PROMPT = self._create_template(template_name = "Code_research_template")
        self.code_query_identification = CODE_QUERY_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()

        CODE_LIST_PROMPT = self._create_template(template_name="Code_generation_list_template")
        self.code_list = CODE_LIST_PROMPT | self.llm | JsonOutputParser()

        self.logger.logger.info("Agent chains initialized")
    
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
        custom_search_tool = CustomSearchTool()
        webscrapper_tool = WebscrapperTool(llm=self.llm)
        generate_code_tool = CodeGeneratorTool(llm=self.llm)
        code_review_tool = CodeReviewTool(llm=self.llm)
        code_corrector_tool = CodeCorrectorTool(llm=self.llm)
        diagram_creator_tool = CodeDiagramCreator(llm=self.llm)

        tools = {
            "web_search": custom_search_tool,
            "web_scrapper": webscrapper_tool,
            "code_generator": generate_code_tool,
            "code_critique": code_review_tool,
            "code_corrector": code_corrector_tool,
            "diagram_creator": diagram_creator_tool
        }
        return tools
    
    def format_scraper_content(self,content):
        """
        Formats content from the web scraper into a consistent text format.
        Handles different possible structures of the scraped content.
        """
        self.logger.logger.debug("Formatting scraper content")
        formatted_text = ""
        
        # Case 1: Content is a list of sections with title and elements
        if isinstance(content, list) and all(isinstance(item, dict) for item in content):
            # Check if it has the section structure
            if any("title" in item and "elements" in item for item in content):
                for section in content:
                    if "title" in section and "elements" in section:
                        title = section.get("title", "")
                        formatted_text += f"\n\n## {title}\n"
                        
                        # Extract and join all text from elements
                        if isinstance(section["elements"], list):
                            section_text = " ".join([
                                elem.get("text", "") 
                                for elem in section["elements"] 
                                if isinstance(elem, dict) and "text" in elem
                            ])
                            formatted_text += section_text
            
            # Case 2: Content is a list of dictionaries with title and summary
            elif any("title" in item and "summary" in item for item in content):
                for item in content:
                    if "title" in item and "summary" in item:
                        title = item.get("title", "")
                        summary = item.get("summary", "")
                        formatted_text += f"\n\n## {title}\n{summary}"

        return formatted_text.strip()
        
    def create_graph(self):
        """Create a LangGraph workflow for agent operations."""
        self.logger.logger.info("Creating agent workflow graph")
        workflow = StateGraph(GraphState)
        logger = self.logger

        # 1. Identify query type
        def identify_query(state: GraphState) -> GraphState:
            node_name = "identify_query"
            node_start_time = logger.log_node_entry(node_name, state)
            
            result = self.question_routing.invoke({"query": state["query"]})
            # logger.logger.info(f"Query identification result: {result}")

            # print(f"result:{result}")
            updated_state = {
                **state,
                "category": result.get("category", []),
                "confidence": result.get("confidence", 0),
                "reasoning": result.get("reasoning", ""),
                "required_tools": result.get("required_tools", [])
            }
            
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

        def code_query_identification(state: GraphState) -> GraphState:
            node_name = "code_query_identification"
            node_start_time = logger.log_node_entry(node_name, state)
            
            logger.logger.info(f"Performing code query identification on: {state['query']}")
            result = self.code_query_identification.invoke({"query": state["query"]})
            print(result)
            updated_state = {**state, "action_type": result.get("action_type", "")}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        # 2. Extract keywords (for web search OR for specific information scraping)
        def extract_keywords(state: GraphState) -> GraphState:
            node_name = "extract_keywords"
            node_start_time = logger.log_node_entry(node_name, state)
            
            query_type = "web_search"
            
            # Check if this is for specific information scraping
            if (state.get("elements_to_retrieve") == "specific_information" and 
                "web_scrape" in state.get("required_tools", [])):
                query_type = "web_scrape"
                
            logger.logger.debug(f"Extracting keywords for query type: {query_type}")
            result = self.question_to_keywords.invoke({
                "query": state["query"],
                "query_type": query_type
            })
            
            updated_state = {**state, "keywords": result.get("keywords", [])}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        # 3. Classify scrape query 
        def classify_scrape_query(state: GraphState) -> GraphState:
            """
            Uses the LLM to classify a web scraping query.
            Determines whether to extract a general summary or specific elements.
            """
            node_name = "classify_scrape_query"
            node_start_time = logger.log_node_entry(node_name, state)
            # logger.logger.info(f"Before classify_scrape_query: {state}")

            result = self.query_web_scrape.invoke({"query": state["query"]})
            # logger.logger.info(f"Web scrape classification result: {result}")
            
            updated_state = {
                **state,
                "category": result.get("category", ""),
                "scrape_url": result.get("url", state.get("scrape_url", "")), 
                "elements_to_retrieve": result.get("elements_to_retrieve", state.get("elements_to_retrieve", []))
            }
            logger.logger.info(f"After classify_scrape_query: {updated_state}")

            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        # 4. Web search handler
        def run_web_search(state: GraphState) -> GraphState:
            node_name = "web_search"
            node_start_time = logger.log_node_entry(node_name, state)
            
            keywords = state.get("keywords", [])
            if not keywords:
                logger.logger.warning("No keywords provided for web search")
                updated_state = {**state, "search_results": []}
                logger.log_node_exit(node_name, updated_state, node_start_time)
                return updated_state

            search_query = " ".join(keywords)
            # logger.logger.info(f"Running web search with query: {search_query}")
            
            # Log the tool call
            logger.log_tool_call("web_search", {"query": search_query})
            search_results = self.tools["web_search"]._run(search_query)
            
            # Log the results
            logger.log_tool_call("web_search", {"query": search_query}, 
                               f"Found {len(search_results)} results")
            
            updated_state = {**state, "search_results": search_results}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        # 5. Web scraper handler
        def run_web_scraper(state: GraphState) -> GraphState:
            """
            Runs web scraping using the classified query data.
            """
            node_name = "web_scrape"
            node_start_time = logger.log_node_entry(node_name, state)
            # logger.logger.info(f"Before web_scrape: {state}")
            url = state.get("scrape_url", "")
            extraction_type = state.get("category", "")
            # logger.logger.info(f"Running web scraper for URL: {url} with extraction type: {extraction_type}")
            
            if extraction_type == "specific_information":
                keywords = state.get("keywords", [])
            else:
                keywords = state.get("elements_to_retrieve", [])

            if not url:
                logger.logger.warning("No valid URL provided for web scraping")
                updated_state = {**state, "scrape_results": [{"error": "No valid URL provided for web scraping"}]}
                logger.log_node_exit(node_name, updated_state, node_start_time)
                return updated_state

            # Log the tool call
            tool_inputs = {"keywords": keywords, "url":url}
            logger.log_tool_call("web_scrapper", tool_inputs)
            
            # Execute the scraping
            raw_content = self.tools["web_scrapper"]._run(keywords=keywords, url=url)
            
            # Format the content
            formatted_content = self.format_scraper_content(raw_content)
            logger.logger.debug(f"Formatted content length: {len(formatted_content)} characters")
            
            updated_state = {**state, "scrape_results": [{"query": state["query"], "content": formatted_content}]}
            # logger.logger.info(f"After web_scrape: {updated_state}")
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        def generate_code(state: GraphState) -> GraphState:
            node_name = "generate_code"
            node_start_time = self.logger.log_node_entry(node_name, state)

            # Call the code generation tool
            code_result = self.tools["code_generator"]._run(query=state["query"],steps=state.get("steps", []))

            updated_state = {**state, "generated_code": code_result}
            self.logger.log_node_exit(node_name, updated_state, node_start_time)
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
        
        # 6. Generate final answer
        def generate_answer(state: GraphState) -> GraphState:
            # Extract scrape results content
            node_name = "generate_answer"
            node_start_time = logger.log_node_entry(node_name, state)
            
            # Extract scrape results content
            scrape_content = ""
            if state.get("scrape_results"):
                for result in state.get("scrape_results", []):
                    if "content" in result:
                        scrape_content = result["content"]
                        break
            
            # Format search results if present
            search_content = json.dumps(state.get("search_results", []), indent=2)
            final_code = json.dumps(state.get("generated_code", []),indent= 2)

            if not final_code:
                prompt = f"""
                Query: {state["query"]}

                {"Search Results: " + search_content if state.get("search_results") else ""}

                {"Scraped Content: " + scrape_content if scrape_content else ""}

                {"Code:" + final_code if state.get("final_code") else ""}

                Based on the information above, provide a comprehensive answer to the query.
                Do not include any citations or source links in your answer - I will add those separately.
                """

                logger.logger.debug("Generating final answer with LLM")
                answer_content = self.llm.invoke(prompt)
            elif final_code:
                logger.logger.debug("Final Answer for code task")
                answer_content = final_code
            
            updated_state = {**state, "final_answer": answer_content}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        def prepare_code_for_critique(state: GraphState) -> GraphState:
            """Updates the state to move the provided code (query) into generated_code."""
            node_name = "prepare_code_for_critique"
            node_start_time = self.logger.log_node_entry(node_name, state)

            updated_state = {**state, "generated_code": state["query"]} 

            self.logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state

        # Add nodes
        workflow.add_node("identify_query", identify_query)
        workflow.add_node("extract_keywords", extract_keywords)
        workflow.add_node("web_search", run_web_search)
        workflow.add_node("classify_scrape_query", classify_scrape_query)
        workflow.add_node("web_scrape", run_web_scraper)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("code_research", code_research)
        workflow.add_node("generate_code", generate_code)
        workflow.add_node("critique_code", critique_code)
        workflow.add_node("code_query_identification", code_query_identification)
        workflow.add_node("prepare_code_for_critique", prepare_code_for_critique)
        workflow.add_node("correct_code", correct_code)
        workflow.add_node("create_diagram", create_diagram)

        # Define edges
        workflow.set_entry_point("identify_query")

        def route_after_identification(state: GraphState) -> List[str]:
            required_tools = state.get("required_tools", [])
            next_nodes = []

            if "web_search" in required_tools:
                next_nodes.append("extract_keywords")  # Web search needs keyword extraction
            if "web_scrape" in required_tools:
                next_nodes.append("classify_scrape_query")  # First classify the web scraping query
            if "code" in required_tools:
                next_nodes.append("code_query_identification")  # Code research needs code execution

            result = next_nodes or ["generate_answer"]
            logger.log_decision("identify_query", result)
            return result

        def route_after_keywords(state: GraphState) -> List[str]:
            if "web_search" in state.get("required_tools", []):
                result = ["web_search"]
            else:
                result = ["generate_answer"]
            logger.log_decision("extract_keywords", result)
            return result
            
        def route_after_scrape_classification(state: GraphState) -> List[str]:
            # logger.logger.info(f"Route after scrape classification : {state}")
            if state.get("category") == "specific_information":
                result = ["extract_keywords"]
            else:
                result = ["web_scrape"]
            logger.log_decision("classify_scrape_query", result)
            return result
        
        def route_code_query_identification(state: GraphState) -> List[str]:
            action_type = state.get("action_type")

            if action_type == "generate":
                return ["generate_code"]
            elif action_type in ["correct", "critique"]: # TO correct the code we first fo through the review process so that it could allow us to correct it based on these feedbacks
                return ["prepare_code_for_critique"]
            elif action_type == "diagram":
                return ["create_diagram"]

        def route_code_loop(state: GraphState) -> List[str]:
            """Loop between generation and critique until the code is valid or iterations run out."""
            action_type = state.get("action_type")
            if state.get("is_code_valid") or state.get("iterations_left", 3) <= 0 or action_type in ["diagram"] :
                return ["generate_answer"]
            return ["correct_code"]
        
        # Connect nodes
        workflow.add_conditional_edges("identify_query", route_after_identification)
        workflow.add_conditional_edges("extract_keywords", route_after_keywords)
        workflow.add_conditional_edges("classify_scrape_query", route_after_scrape_classification)
        workflow.add_conditional_edges("critique_code", route_code_loop)
        workflow.add_conditional_edges("code_query_identification", route_code_query_identification)
        
        # Add edges
        workflow.add_edge("code_research","generate_code")
        workflow.add_edge("generate_code", "critique_code")
        workflow.add_edge("correct_code", "critique_code")
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("web_scrape", "generate_answer")
        workflow.add_edge("prepare_code_for_critique", "critique_code")
        # workflow.add_edge("final_code_output", "generate_answer")
        # Set the output
        workflow.set_finish_point("generate_answer")

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
            # Store the results in a dictionary
            final_result = {
                "query": result.get("query"),
                "generated_code": result.get("generated_code"),
                "is_code_valid": result.get("is_code_valid"),
                "iterations_left": result.get("iterations_left"),
                "critique_feedback": result.get("critique_feedback"),
                "final_answer": result.get("final_answer")
            }

            self.logger.end_agent_run(result)
            return result.get("final_answer", "No answer generated.")
        except Exception as e:
            self.logger.log_error("agent_run", e)
            self.logger.logger.error(f"Agent run failed: {str(e)}")
            return f"Error running agent: {str(e)}"


    


