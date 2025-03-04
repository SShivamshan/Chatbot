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

# SOLUTION adopted from : https://medium.com/@sahin.samia/how-to-build-a-interactive-personal-ai-research-agent-with-llama-3-2-b2a390eed63e 
# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot 
class GraphState(TypedDict):
    query: str
    category: Optional[List[str]]
    confidence: Optional[float]
    reasoning: Optional[str]
    required_tools: Optional[List[str]]
    keywords: Optional[List[str]]
    search_results: Optional[List[Dict[str, Any]]]
    scrape_results: Optional[List[Dict[str, Any]]]
    final_answer: Optional[str]
    final_json: Optional[Dict[str, Any]]

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
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        context_length: int = 18000,
        chatbot: Optional[Any] = None
    ):
        """Initialize the agent with modern LangChain patterns."""
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            context_length=context_length
        )
        # Initialize LLM
        self.llm = chatbot if chatbot else Chatbot(
            base_url=base_url,
            model=model_name,
            context_length=context_length
        )
        # Initialize components
        self.tools = self.initialize_tools()
        self.ai_template = load_ai_template('config/config.yaml')
        
        # Initialize agent chains
        QUERY_IDENTIFICATION_PROMPT = self._query_identification_template()
        self.question_routing = QUERY_IDENTIFICATION_PROMPT | self.llm | JsonOutputParser()
        
        QUERY_WEB_PROMPT = self._query_web_template()
        self.question_to_keywords = QUERY_WEB_PROMPT | self.llm | JsonOutputParser()

        QUERY_WEB_SCRAPE_CLASSIFICATION_PROMPT = self._query_web_scrape_template()
        self.query_web_scrape = QUERY_WEB_SCRAPE_CLASSIFICATION_PROMPT | self.llm | JsonOutputParser()
        
    def _query_identification_template(self):
        try:
            templates = load_ai_template(config_path="config/config.yaml")
            template_config = templates["Agent_templates"]["Query_identification_template"]
            template = template_config["template"]
            input_variables = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize query identification: {str(e)}")
    
    def _query_web_template(self):
        try:
            templates = load_ai_template(config_path="config/config.yaml")
            template_config = templates["Agent_templates"]["Query_web_template"]
            template = template_config["template"]
            input_variables = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize query web: {str(e)}")
        
    def _query_web_scrape_template(self):
        try:
            templates = load_ai_template(config_path="config/config.yaml")
            template_config = templates["Agent_templates"]["Query_web_scrape_classification"]
            template = template_config["template"]
            input_variables = [var["name"] for var in template_config.get("input_variables", [])]
            return PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize web scrape query classification: {str(e)}")

    
    def initialize_tools(self):
        # Instantiate the CustomSearchTool
        custom_search_tool = CustomSearchTool()
        # Instantiate the WebscrapperTool
        webscrapper_tool = WebscrapperTool()
        tools = {
            "web_search": custom_search_tool,
            "web_scrapper": webscrapper_tool
        }
        return tools
        
    def create_graph(self):
        """Create a LangGraph workflow for agent operations."""
        workflow = StateGraph(GraphState)

        # 1. Identify query type
        def identify_query(state: GraphState) -> GraphState:
            result = self.question_routing.invoke({"query": state["query"]})
            return {
                **state,
                "category": result.get("category", []),
                "confidence": result.get("confidence", 0),
                "reasoning": result.get("reasoning", ""),
                "required_tools": result.get("required_tools", [])
            }

        # 2. Extract keywords (ONLY for web search, NOT for web scraping)
        def extract_keywords(state: GraphState) -> GraphState:
            if "web_scraper" in state.get("required_tools", []):
                return state  # Skip keyword extraction for scraping

            result = self.question_to_keywords.invoke({
                "query": state["query"],
                "query_type": "web_search"
            })
            return {**state, "keywords": result.get("keywords", [])}

        def classify_scrape_query(state: GraphState) -> GraphState:
            """
            Uses the LLM to classify a web scraping query.
            Determines whether to extract a general summary or specific elements.
            """
            result = self.query_web_scrape.invoke({"query": state["query"]})

            return {
                **state,
                "category": result.get("category", ""),
                "scrape_url": result.get("url", ""),
                "elements_to_retrieve": result.get("elements_to_retrieve", [])
            }

        # 3. Web search handler (Uses extracted keywords)
        def run_web_search(state: GraphState) -> GraphState:
            keywords = state.get("keywords", [])
            if not keywords:
                return {**state, "search_results": []}

            search_query = " ".join(keywords)
            search_results = self.tools["web_search"].run(search_query)
            return {**state, "search_results": search_results}

        # 4. Web scraper handler (Runs immediately after query identification)
        def run_web_scraper(state: GraphState) -> GraphState:
            """
            Runs web scraping using the classified query data.
            """
            url = state.get("scrape_url", "")
            keywords = state.get("elements_to_retrieve", [])

            if not url:
                return {**state, "scrape_results": [{"error": "No valid URL provided for web scraping"}]}

            content = self.tools["web_scrapper"].run({
                "url": url,
                "keywords": keywords  # Extract specific semantic information
            })

            return {**state, "scrape_results": [{"query": state["query"], "content": content}]}

        # 5. Generate final answer
        def generate_answer(state: GraphState) -> GraphState:
            prompt = f"""
            Query: {state["query"]}

            Search Results: {json.dumps(state.get("search_results", []), indent=2)}

            Scraped Content: {json.dumps(state.get("scrape_results", []), indent=2)}

            Based on the information above, provide a comprehensive answer to the query.
            Do not include any citations or source links in your answer - I will add those separately.
            """

            answer_content = self.llm.invoke(prompt)
            return {**state, "final_answer": answer_content}

        # Add nodes
        workflow.add_node("identify_query", identify_query)
        workflow.add_node("extract_keywords", extract_keywords)
        workflow.add_node("web_search", run_web_search)
        workflow.add_node("classify_scrape_query", classify_scrape_query)
        workflow.add_node("web_scraper", run_web_scraper)
        workflow.add_node("generate_answer", generate_answer)

        # Define edges
        workflow.set_entry_point("identify_query")

        def route_after_identification(state: GraphState) -> List[str]:
            required_tools = state.get("required_tools", [])
            next_nodes = []

            if "web_search" in required_tools:
                next_nodes.append("extract_keywords")  # Web search needs keyword extraction
            if "web_scraper" in required_tools:
                next_nodes.append("classify_scrape_query")  # First classify the web scraping query

            return next_nodes or ["generate_answer"]

        def route_after_keywords(state: GraphState) -> List[str]:
            return ["web_search"] if "web_search" in state.get("required_tools", []) else ["generate_answer"]

        def route_after_search(state: GraphState) -> List[str]:
            return ["generate_answer"]
        def route_after_scrape_classification(state: GraphState) -> List[str]:
            return ["web_scraper"]
        def route_after_web_scraper(state: GraphState) -> List[str]:
            return ["generate_answer"]
        
        # Connect nodes
        workflow.add_conditional_edges("identify_query", route_after_identification)
        workflow.add_conditional_edges("extract_keywords", route_after_keywords)
        workflow.add_conditional_edges("web_search", route_after_search)
        workflow.add_conditional_edges("classify_scrape_query", route_after_scrape_classification)
        workflow.add_conditional_edges("web_scraper", route_after_web_scraper)
        workflow.add_edge("web_scraper", "generate_answer")
        workflow.add_edge("web_search", "generate_answer")

        # Set the output
        workflow.set_finish_point("generate_answer")

        return workflow

    
    def initialize_agent(self):
        """Initialize the agent with LangGraph."""
        # Create the graph
        graph = self.create_graph()
        
        # Compile the graph
        executor = graph.compile()
        
        return executor
    
    def run(self, query: str):
        """Run the agent with the given query."""
        executor = self.initialize_agent()
        result = executor.invoke({"query": query})
        return result.get("final_answer", "No answer generated.")





