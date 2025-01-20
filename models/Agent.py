import yaml
from typing import List, Optional, Dict,Any
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.schema import BaseRetriever
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description_and_args
from models.Model import Chatbot
from src.utils import SearchTool,CitationTool

def load_ai_template(template_name: str) -> Dict:
    with open(template_name, 'r') as file:
        config = yaml.safe_load(file)
    return config['ai_templates']


class Agent(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:3b")
    context_length: int = Field(default=18000)
    tools: List[Tool] = Field(default_factory=list)
    ai_template: dict = Field(default_factory=dict)
    agent: Optional[AgentExecutor] = Field(default=None, exclude=True)
    vector_store: Optional[BaseRetriever] = Field(default=None, exclude=True)
    memory: Optional[ConversationBufferMemory] = Field(default=None, exclude=True)
    llm: Optional[Any] = Field(default=None, exclude=True)

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        context_length: int = 18000,
        vector_store: Optional[BaseRetriever] = None,
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
        self.vector_store = vector_store
        self.tools = self.initialize_tools()
        self.ai_template = self._load_ai_template('config/config.yaml')
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="full_input"  # Set input_key to align with combined input
        )

    def initialize_tools(self) -> List[Tool]:
        """Initialize tools with proper error handling."""
        if not self.vector_store:
            raise ValueError("Vector store must be initialized before creating tools")

        return [
            Tool(
                name="Search",
                func=SearchTool(retriever=self.vector_store)._run,
                description="Search for relevant information in the knowledge base."
            ),
            Tool(
                name="GenerateCitation",
                func=CitationTool(retriever=self.vector_store)._run,
                description="Generate a citation from the relevant PDF content."
            )
        ]

    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """Create the agent prompt with proper tool formatting."""
        template = """Respond to the human as helpfully and accurately as possible. 
        You have access to the following tools:
        {tools}

        Valid "action" values: "Final Answer" or {tool_names}
        Always use the following JSON format for actions:
        ```
        {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }}
        ```

        Respond directly if appropriate. Input: {input}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{full_input}"),
        ])

        prompt.partial(
            tools=render_text_description_and_args(list(self.tools)),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        return prompt

    def initialize_agent(self) -> None:
        """Initialize the agent with properly formatted tools."""
        if self.agent is None:
            # Create prompt with tool information
            prompt = self._create_agent_prompt()

            chain = (
                RunnablePassthrough.assign(
                    agent_scratchpad=lambda x: format_log_to_str(x.get("intermediate_steps", {})),
                    chat_history=lambda x: self.memory.chat_memory.messages,
                )
                | prompt
                | self.llm
                | JSONAgentOutputParser()
            )

            self.agent = AgentExecutor(
                agent=chain,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                max_iterations=5,
                handle_parsing_errors=True
            )

    async def arun(self, query: str) -> str:
        """Async run method with improved error handling."""
        try:
            self.initialize_agent()

            combined_input = {
                "tools": "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools),
                "input": query,
                "agent_scratchpad": format_log_to_str({}),
                "tool_names": ", ".join(tool.name for tool in self.tools),
            }

            response = await self.agent.ainvoke({"full_input": combined_input})
            return response["output"]
        except Exception as e:
            raise Exception(f"Async agent execution failed: {str(e)}")

    def run(self, query: str) -> str:
        """Synchronous run method with improved error handling."""
        try:
            self.initialize_agent()

            combined_input = {
                "tools": "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools),
                "input": query,
                "agent_scratchpad": format_log_to_str({}),
                "tool_names": ", ".join(tool.name for tool in self.tools),
            }

            response = self.agent.invoke({"full_input": combined_input})
            return response["output"]
        except Exception as e:
            raise Exception(f"Agent execution failed: {str(e)}")


    def _load_ai_template(self, config_path: str) -> dict:
        """Load AI template with error handling."""
        try:
            return load_ai_template(config_path)
        except Exception as e:
            raise ValueError(f"Failed to load AI template: {str(e)}")
