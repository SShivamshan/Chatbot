# Multi-Agent System
A comprehensive multi-agent framework designed to handle various tasks through specialized AI agents, including document processing, code generation, web search, and knowledge base management.

## Architecture  
This system implements a multi-agent architecture with specialized agents working together under supervision to handle complex tasks efficiently. The framework leverages intelligent orchestration to automatically delegate tasks to the most appropriate agents based on task requirements.

![Agent Diagram](/images/Agent_structure.png)

### Core Components

- SupervisorAgent.py - Central orchestrator that coordinates and manages other agents
- AgenticRAG.py - Advanced Retrieval-Augmented Generation system combining PDF Agent and Web Agent capabilities for intelligent information retrieval from both local documents and online sources
- Model.py – Defines the core model interface and manages configuration for LLM integration across all agents. It includes two main classes:
    - One subclassing ChatOllama for Ollama-based models.
    - One subclassing ChatOpenAI for OpenAI API-based models.

### Specialized Agents

- PDFAgent.py - Specialized in processing, extracting, and analyzing PDF documents(online or local)
- CodeAgent.py - Handles code generation, analysis, and programming-related tasks
- WebAgent.py - Performs web search, scraping, and online information gathering
- KBAgent.py - Knowledge base management and querying capabilities

### Supporting Modules

- RAG.py - Retrieval-Augmented Generation implementation for enhanced context awareness, utilizing ChromaDB for vector storage and OLLAMA embeddings for semantic search

## Features

- Multi-Agent Coordination: Intelligent task distribution and agent orchestration
- Document Processing: Advanced PDF analysis and content extraction
- Code Intelligence: Automated code generation, review, and optimization
- Web Integration: Real-time web search and data collection
- Knowledge Management: Efficient storage and retrieval of information
- RAG Implementation: Context-aware responses using retrieval-augmented generation

## Running agent on the command line


## Observation 

- A key observation is that the quality of responses from the Agent system using the OpenAI API is significantly better compared to local models.
- The `nomic-embed-text` embeddings perform reliably with both OpenAI and LLaMA-based models.
- Local models often struggle with template adherence, producing outputs that do not consistently follow instructions or formatting requirements. In contrast, OpenAI models handle structured outputs and instruction-following much more effectively. 