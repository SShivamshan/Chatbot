# Multi-Agent System
A comprehensive multi-agent framework designed to handle various tasks through specialized AI agents, including document processing, code generation, web search, and knowledge base management.

## Architecture  
This system implements a multi-agent architecture with specialized agents working together under supervision to handle complex tasks efficiently. The framework leverages intelligent orchestration to automatically delegate tasks to the most appropriate agents based on task requirements.

![Agent Diagram](/images/Agent_structure.png)

### Core Components

- SupervisorAgent.py - Central orchestrator that coordinates and manages other agents
- AgenticRAG.py - Advanced Retrieval-Augmented Generation system combining PDF Agent and Web Agent capabilities for intelligent information retrieval from both local documents and online sources
- Model.py - Core model interface and configuration management for LLM integration used by all agents

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

## Observation 
