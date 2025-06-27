# LLM Chatbot Project

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Features](#features)
6. [Dependencies](#dependencies)
7. [Configuration](#configuration)
8. [Contributing](#contributing)
9. [License](#license)
10. [Troubleshooting](#troubleshooting)
11. [Improvements](#improvements)

## Introduction

This project implements a chatbot application using Large Language Models (LLM). The chatbot is designed to interact with users, process their inputs, and provide relevant responses. It includes features for handling documents, images, and schematics, with integration with ChromaDB for efficient storage and retrieval. FInally plans are in place to create a Machine learning chat bot type sessions that allows users to interact with their machine learning use cases and understand them better. Currently, there are three forms of application, a simple chat bot, a RAG module and finally an AGENT capable of interacting with PDF and research more relevant information on the web. Currently a Work in Progress especially the Agent part. 

## Project Structure
The chatbot will start, and you can begin interacting with it through the command line interface.
```
llm/
├── app.py                  # Main application entry point
├── config/
│   └── config.yaml        # Template configurations 
│
├── database/              # Database related files
│   ├── app.db
│   └── chroma.sqlite3
│
├── images/
│   └── ai.png            # Project images
│
├── models/               # Model definitions
│   ├── Agent.py         # Agent model implementation
│   ├── Model.py         # Base model implementation
│   └── RAG.py           # RAG model implementation
│
├── pages/               # Application pages
│   ├── account.py       # Account management page
│   ├── base.py         # Base page template for the app.db 
│   ├── chat.py         # Chat interface renderer
│   ├── data_db.py      # Database interface containing the chat session, image and table management
│   ├── history.py      # Chat history page renderer
│   └── home.py         # Home page
│
├── src/        
|   ├── download_tokens.py  # Allows to download tokens     
│   ├── main.py             # Main logic
│   └── utils.py            # Utility functions
│
├── LICENSE             # License file
└──  README.md          # Project documentation
```

## Installation

1. Clone the repository:  git clone https://github.com/yourusername/llm-chatbot.git

2. Create a virtual environment:
``
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
``
or 
``conda create -n venv python = 3.10(example) && 
conda activate venv 
``

3. Install the required dependencies: `` pip install -r requirements.txt``


## Usage

To run the chatbot application, execute the following command from the project root: ``streamlit run src/main.py``
## Features

- Interactive chatbot using LLMs mostly using Ollama (llama3.2 : 3 billion model), llava:7b for images cases. 
- Document processing and analysis
- Image and schematic handling
- Integration with ChromaDB for efficient storage
- Agents that does the same things as RAG but with the ability to run a search on the web through keywords. 
- Further addition of machine learning and chat bot mix
- Offloads the model from the GPU when switching between sessions or logging out. 
- Possibility of running the agent in the command line by running the run_agent python file. 

## Dependencies

- Python                3.7+
- Streamlit             1.41.0
- Ollama                0.5.1
- Langchain             0.3.9 
- Langgraph             0.3.2
- Langchain-chroma      0.2.0
- Langchain-community   0.3.9

Since we use ``unstructured`` module directly, it requires the installation of dependencies which are quiet troublesome. 
Since the application is primarily based on PDF files, we need to install the following dependencies : 

    poppler-utils tesseract-ocr libmagic-dev

### Installing Dependencies with Conda

While poppler-utils can be installed via pip, we recommend using Conda for the other dependencies to avoid compatibility issues:

    conda install -c conda-forge libmagic

    conda install conda-forge::tesseract

For a complete list of dependencies, refer to the `requirements.txt` file.

## Configuration

Any configuration such as the temperature of the models, the possibility of adding a new ollama model. 

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
The application requires specific NLTK tokenizers for text processing. We've included a function that automatically checks for and installs required tokenizers if they're missing.
If you encounter tokenizer-related errors, you can manually install them using:

    python -m ntlk.downloader name of the tokenizer or tagger 

Example :   `` python -m ntlk.downloader punk_tab ``

## Improvements

1. Right now the pdf can only be added one by one and can only render one pdf at the time. Tried to use st.tabs to render the pdf per tabs but only the first tab is getting rendered. 
2. Takes some time to chunks the pdf when using unstructured module especially true when retreiving tables and images. 
3. Perhaps add some javascript components. 
4. Improvements linked to constrained environments. 
5. Possibly adding docker support 
6. Possibility of adding the configuration to interact with google applications especially for code like a sandbox application where we could run a test file.  
7. Modular Coding Agent Architecture : To improve the structure and performance of the coding agent, consider dividing it into three specialized sub-agents, each with a distinct responsibility:

    - Research Agent: This agent is responsible for researching the problem domain. It can also explore and analyze the folder and file structure to gather relevant context before any coding begins.
    - Coder Agent: This agent plans and writes the code based on the insights provided by the research agent. It defines the overall architecture, selects tools or libraries, and implements the solution.
    - Reviewer Agent: This agent reviews and executes the code in a controlled, sandboxed environment. It captures errors, evaluates output, and provides structured feedback to improve the implementation iteratively.
