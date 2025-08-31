# LLM Chatbot Project

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.41.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Quick start](#quick-start)
4. [Features](#features)
5. [Dependencies](#dependencies)
6. [Contributing](#contributing)
7. [License](#license)
8. [Troubleshooting](#troubleshooting)
9. [Improvements](#improvements)

## Introduction

This project implements a chatbot application using Large Language Models (LLMs). The chatbot is designed to interact with users, process their inputs, and provide relevant responses. It includes capabilities for handling documents, images, and schematics, with integration with ChromaDB for efficient storage and retrieval.

Plans are also in place to develop a machine learning chatbot that allows users to interact with their ML use cases and gain deeper insights. Currently, the project features three application forms:

- Simple Chatbot: Handles basic conversational interactions.
- RAG (Retrieval-Augmented Generation) Module: Provides enhanced responses by retrieving relevant context from Vector DBs. 
- Agent: Capable of interacting with PDFs and researching relevant information on the web and various other tasks. Please use the README inside the models folder

All three applications: chatbot, RAG, and agent function properly with offline models as well as API-based models such as OpenAI’s GPT-4.1 and other multi modal models, which currently outperforms the local models.

This project served as a hands-on exploration of LangChain tools and agentic architectures. While the current implementation works well, there is room for improvements in both code quality and user interface[Improvements](#improvements) section. 

## Project Structure
The chatbot will start, and you can begin interacting with it through the command line interface.
```
llm/
├── app.py                  # Entry point for running the application
│
├── config/
│   └── template.yaml       # Centralized file containing all the template used in this project
│
├── database/
│   ├── app.db              # Application-specific database
│   └── chroma.sqlite3      # ChromaDB storage for embeddings
│
├── images/                 # Visual assets and output images
│   ├── Agent_structure.png # Diagram showing agent architecture
│   ├── ai.png              # General AI-related image asset
│   └── img_output/         # Folder for generated image outputs
│
├── LICENSE                 # Project license
│
├── models/                 # Core logic for LLM agents and integrations
│   ├── AgenticRAG.py       # Agent combining RAG with agentic workflows
│   ├── CodeAgent.py        # Agent specialized in coding tasks
│   ├── KBAgent.py          # Knowledge base agent for structured queries
│   ├── Model.py            # Core model interface (ChatOllama, ChatOpenAI)
│   ├── PDFAgent.py         # Agent for processing PDFs
│   ├── RAG.py              # Retrieval-Augmented Generation module
│   ├── README.md           # Documentation for models folder
│   ├── SupervisorAgent.py  # Agent managing/overseeing other agents
│   └── WebAgent.py         # Agent capable of web search & retrieval
│
├── pages/                  # UI/Frontend pages (likely Streamlit or similar)
│   ├── account.py          # User account management
│   ├── base.py             # Base layout/components
│   ├── chat.py             # Chat interface
│   ├── data_db.py          # Database interaction page
│   ├── history.py          # Chat history view
│   ├── home.py             # Landing page
│
├── src/                    # Utility scripts and application logic
│   ├── AgentLogger.py      # Logging utilities for agent workflows
│   ├── download_tokens.py  # Script for handling token downloads
│   ├── main.py             # Main execution script
│   ├── README.md           # Documentation for src folder
│   ├── run_agent.py        # CLI entry for running an agent directly
│   └── utils.py            # Helper functions/utilities
│
└── README.md               # Main project documentation
```

## Quick start
```
1. Clone the repository
git clone https://github.com/yourusername/Chatbot.git
cd llm-chatbot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
./src/entrypoint.sh

# 5. Run the app
streamlit run src/main.py
```

Right now, we have a sequential execution problem where we need to manually start the server first, then run the Streamlit application separately. This happens because:
- Streamlit is blocking/sequential: When streamlit run executes, it blocks the current thread and doesn't return control to continue executing subsequent code
- Manual coordination required: Users must remember to start the server in one terminal, then run the Streamlit app in another terminal
- Poor user experience: This creates friction and potential for errors (forgetting to start the server, wrong order, etc.)

## Features

- **Interactive Chatbot with LLMs**: Primarily built using Ollama (LLaMA 3.2 – 3B model) for general conversations, LLaVA-7B for image-related tasks, and the OpenAI API for enhanced performance.
- **Document Processing & Analysis**: Supports efficient handling and analysis of text-based documents.
- **Image & Schematic Handling**: Capable of interpreting and interacting with visual data.
- **ChromaDB Integration**: Provides efficient vector storage and retrieval for contextual responses.
- **Agent System**: Extends RAG functionality with the ability to perform keyword-based web searches and execute additional automated tasks.
- **Machine Learning Integration**: Plans to combine chatbot interactions with ML workflows for deeper insights into use cases.
- **Resource Optimization**: Automatically offloads models from the GPU when switching sessions or logging out.
- **Flexible Execution**: Agents can also be run directly from the command line via the `run_agent.py` script. 

## Dependencies

- Python                3.7+
- Streamlit             1.41.0
- Ollama                0.5.1
- Langchain             0.3.9 
- Langgraph             0.3.2
- Langchain-chroma      0.2.0
- Langchain-community   0.3.9
- Langchain-openai      0.3.31
- Langchain-tavily      0.2.4


Since we use ``unstructured`` module directly, it requires the installation of dependencies which are quiet troublesome. 
Since the application is primarily based on PDF files, we need to install the following dependencies : 

    poppler-utils tesseract-ocr libmagic-dev

### Installing Dependencies with Conda

While poppler-utils can be installed via pip, we recommend using Conda for the other dependencies to avoid compatibility issues:

    conda install -c conda-forge libmagic

    conda install conda-forge::tesseract

For a complete list of dependencies, refer to the `requirements.txt` file.


## Contributing

Contributions to this project are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting
### NVIDIA GPU Compatibility
If you're using an NVIDIA GPU, you might encounter compatibility issues with PyTorch versions that are installed as dependencies. This typically occurs because the installed torch/torchvision versions might not match your GPU's CUDA version.

Solution: Install PyTorch with one version lower than your current NVIDIA CUDA version.

Example: If you have CUDA 12.2, install PyTorch for CUDA 12.1

### DuckDuckGo search to Tavily API
It's preferable to install this version of duckduckgo_search==6.3.5, if the latest version lauches the following error : `duckduckgo_search.exceptions.RatelimitException: https://duckduckgo.com/ 202 Ratelimit` 

Thus run the following command to install duckduckgo : ``pip3 install -U duckduckgo_search==6.3.5``
Solution found : https://github.com/open-webui/open-webui/discussions/6624 

Due to persistent rate limit issues and the unreliability of DuckDuckGo for more robust search requirements, I transitioned to the [Tavily Search API] https://docs.tavily.com/documentation/api-reference/endpoint/search for a more scalable and consistent solution. To integrate Tavily with LangChain, follow this guide: : https://python.langchain.com/docs/integrations/tools/tavily_search/ 

### NLTK Tokenizers
The application requires specific NLTK tokenizers for text processing. I've included a function that automatically checks for and installs required tokenizers if they're missing.
If you encounter tokenizer-related errors, you can manually install them using:

    python -m ntlk.downloader name of the tokenizer or tagger 

Example :   `` python -m ntlk.downloader punk_tab ``


## Improvements

1. Right now the pdf can only be added one by one and can only render one pdf at the time. Tried to use st.tabs to render the pdf per tabs but only the first tab is getting rendered. 
2. Takes some time to chunks the pdf when using unstructured module especially true when retreiving tables and images. 
3. Perhaps add some javascript components to create a better render. 
4. Improvements linked to constrained environments. 
5. Possibly adding docker support 
6. Possibility of adding the configuration to interact with google applications especially for code like a sandbox application where we could run a test file.  
7. The agents's graphstate/logs inside the app get's updated through a primitive approach but could be enhanced. 
8. Modular Coding Agent Architecture : To improve the structure and performance of the coding agent, consider dividing it into three specialized sub-agents, each with a distinct responsibility:

    - Research Agent: This agent is responsible for researching the problem domain. It can also explore and analyze the folder and file structure to gather relevant context before any coding begins.
    - Coder Agent: This agent plans and writes the code based on the insights provided by the research agent. It defines the overall architecture, selects tools or libraries, and implements the solution.
    - Reviewer Agent: This agent reviews and executes the code in a controlled, sandboxed environment. It captures errors, evaluates output, and provides structured feedback to improve the implementation iteratively.
