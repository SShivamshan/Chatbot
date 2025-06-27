from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
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
    scrape_category: Optional[str]
    scrape_results: Optional[List[Dict[str, Any]]]
    scrape_url: Optional[str]
    final_answer: Optional[str]
    sources: Optional[List]

class WebAgent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: Dict = Field(default_factory=dict)
    llm: Optional[Any] = Field(default=None, exclude=True)
    question_routing: Optional[Any] = Field(default=None, exclude=True)
    question_to_keywords: Optional[Any] = Field(default=None, exclude=True)
    query_web_scrape: Optional[Any] = Field(default=None, exclude=True)
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
        self.logger = AgentLogger(log_level=log_level, pretty_print=pretty_print,Agent_name="Web Agent")
        self.logger.logger.info(f"Initializing WebAgent with model: {model_name}")

        # Initialize LLM
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )
        self.logger.logger.info("Web Agent LLM initialized")
        # Initialize components
        self.tools = self.initialize_tools()
        QUERY_IDENTIFICATION_PROMPT = self._create_template(template_name="Web_type_identification_template")
        self.question_routing = QUERY_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()

        QUERY_WEB_PROMPT = self._create_template(template_name="Query_web_template")
        self.question_to_keywords = QUERY_WEB_PROMPT | self.llm | JsonOutputParser()

        QUERY_WEB_SCRAPE_CLASSIFICATION_PROMPT = self._create_template(template_name="Query_web_scrape_classification")
        self.query_web_scrape = QUERY_WEB_SCRAPE_CLASSIFICATION_PROMPT | self.llm | JsonOutputParser()

        self.logger.logger.info("Web Agent Tools and templates loaded")
        
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

        tools = {
            "web_search": custom_search_tool,
            "web_scrapper": webscrapper_tool,
        }
        return tools
    

    def format_scraper_content(self,content):
        """
        Formats content from the web scraper into a consistent text format.
        Handles different possible structures of the scraped content.
        """
        self.logger.logger.debug("Formatting scraper content")
        formatted_text = ""

        # Case 1: Content is a list of elements corresponding to the matched keywords
        if isinstance(content, list):
            for section_text in content:
                formatted_text += section_text
            
        # Case 2: Content is a list of dictionaries with title and summary
        elif isinstance(content, Dict):
                title = content.get("title", "")
                summary = content.get("summary", "")
                formatted_text += f"\n\n## {title}\n{summary}"

        return formatted_text.strip()
    
    def create_graph(self):
        """Create a workflow for Code agent operations."""
        self.logger.logger.info("Creating WebAgent workflow graph")
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
        
        # 2. Extract keywords (for web search OR for specific information scraping)
        def extract_keywords(state: GraphState) -> GraphState:
            node_name = "extract_keywords"
            node_start_time = logger.log_node_entry(node_name, state)
            
            query_type = "web_search"
            
            # Check if this is for specific information scraping
            if (state.get("scrape_category") == "specific_information" and 
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
                "scrape_category": result.get("category", ""),  
                "scrape_url": result.get("url", state.get("scrape_url", ""))
            }

            logger.logger.info(f"After classify_scrape_query: {updated_state}")

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
                keywords = []

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
            logger.logger.info(f"Formatted content length: {len(formatted_content)} characters")
            
            updated_state = {**state, "scrape_results": [{"query": state["query"], "content": formatted_content}]}
            # logger.logger.info(f"After web_scrape: {updated_state}")
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        # 4. Web search handler
        def run_web_search(state: GraphState) -> GraphState:
            node_name = "web_search"
            node_start_time = logger.log_node_entry(node_name, state)
            
            keywords = state.get("keywords", [])
            if not keywords:
                logger.logger.warning("No keywords provided for web search using query directly")
                search_results = self.tools["web_search"]._run(keywords)
                # Log the results
                logger.log_tool_call("web_search", {"query": state.get("query")}, 
                                f"Found {len(search_results)} results")
                
                updated_state = {**state, "search_results": search_results}
            else:
                logger.logger.info(f"Running web search with using keywords")
                # Log the tool call
                logger.log_tool_call("web_search", {"query": keywords})
                search_results = self.tools["web_search"]._run(keywords)
                # Log the results
                logger.log_tool_call("web_search", {"query": keywords}, 
                                f"Found {len(search_results)} results")
                
                updated_state = {**state, "search_results": search_results}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
        
        # 6. Generate final answer
        def generate_answer(state: GraphState) -> GraphState:
           
            node_name = "generate_answer"
            sources = []
            node_start_time = logger.log_node_entry(node_name, state)
            scrape_content = ""
            if state.get("scrape_results"):
                for result in state["scrape_results"]:
                    if "content" in result:
                        scrape_content = result["content"]
                        break

            if state.get("scrape_category") == "general_summary" and scrape_content:
                final_answer = scrape_content

            elif state.get("search_results"):
                search_results = state["search_results"]
                top_results = []
                all_images = []

                if isinstance(search_results, dict):
                    top_results = search_results.get("results", [])
                    all_images = search_results.get("images", [])
                elif isinstance(search_results, list):
                    for response in search_results:
                        top_results.extend(response.get("results", []))
                        all_images.extend(response.get("images", []))

                # Format search results for LLM
                result_text = "\n\n".join([
                    f"Title: {res.get('title')}\nContent: {res.get('content')}"
                    for res in top_results if res.get("title") and res.get("content")
                ])

                # image entries with url + description
                formatted_images = [
                    {"url": img.get("url"), "description": img.get("description")}
                    for img in all_images if img.get("url") and img.get("description")
                ]

                # Build prompt
                prompt = f"""
                Query: {state['query']}
                
                Search Results:
                {result_text}

                Based on the above search results, provide a comprehensive summary for the given query.
                """

                final_answer = self.llm.invoke(prompt)

                # Prepare sources
                sources = [
                    {"title": res.get("title"), "url": res.get("url")}
                    for res in top_results if res.get("title") and res.get("url")
                ]

                # Append images as sources
                sources.extend(formatted_images)
            else:
                prompt = f"""
                Query: {state["query"]}
                {"Scraped Content: " + scrape_content if scrape_content else ""}

                Based on the information above, provide a comprehensive answer to the query.
                Do not include any citations or source links in your answer - I will add those separately.
                """
                final_answer = self.llm.invoke(prompt)

            updated_state = {**state, "final_answer": final_answer, "sources": sources}
            logger.log_node_exit(node_name, updated_state, node_start_time)
            return updated_state
    
        # Add nodes
        workflow.add_node("identify_query", identify_query)
        workflow.add_node("extract_keywords", extract_keywords)
        workflow.add_node("web_search", run_web_search)
        workflow.add_node("classify_scrape_query", classify_scrape_query)
        workflow.add_node("web_scrape", run_web_scraper)
        workflow.add_node("generate_answer", generate_answer)

        # Define start point
        workflow.set_entry_point("identify_query")

        def route_after_identification(state: GraphState) -> List[str]:
            required_tools = state.get("required_tools", [])
            next_nodes = []

            if "web_search" in required_tools:
                next_nodes.append("extract_keywords")  # Web search needs keyword extraction
            if "web_scrape" in required_tools:
                next_nodes.append("classify_scrape_query")  # First classify the web scraping query

            result = next_nodes or ["generate_answer"]
            logger.log_decision("identify_query", result)
            return result
        
        def route_after_keywords(state: GraphState) -> List[str]:
            if "web_search" in state.get("required_tools", []):
                result = ["web_search"]
            elif "web_scrape" in state.get("required_tools",[]):
                result = ["web_scrape"]
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
        
        # Connect nodes
        workflow.add_conditional_edges("identify_query", route_after_identification)
        workflow.add_conditional_edges("extract_keywords", route_after_keywords)
        workflow.add_conditional_edges("classify_scrape_query", route_after_scrape_classification)
        
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("web_scrape", "generate_answer")
        workflow.set_finish_point("generate_answer")

        return workflow
    
    def initialize_agent(self):
        """Initialize the agent with LangGraph."""
        self.logger.logger.info("Initializing WebAgent workflow")
        # Create the graph
        graph = self.create_graph()
        
        # Compile the graph
        self.logger.logger.info("Compiling WebAgent workflow graph")
        executor = graph.compile()
        
        return executor
    
    def run(self, query: str):
        """Run the agent with the given query."""
        self.logger.start_agent_run(query)
        self.logger.logger.info(f"Running WebAgent with query: {query}")
        
        try:
            executor = self.initialize_agent()
            result = executor.invoke({"query": query})

            self.logger.end_agent_run(result)
            return { # the final answer can be in AI message type or just str
                "answer": result.get("final_answer", "No answer generated.").content if isinstance(result.get("final_answer", "No answer generated."), AIMessage) else result.get("final_answer", "No answer generated."),
                "sources": result.get("sources", []),
                "query": result.get("query", "")
            }
        except Exception as e:
            self.logger.log_error("agent_run", e)
            self.logger.logger.error(f"WebAgent run failed: {str(e)}")
            return f"Error running WebAgent: {str(e)}"