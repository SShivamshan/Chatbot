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
10. [Improvements](#improvements)

## Introduction

This project implements a chatbot application using Large Language Models (LLM). The chatbot is designed to interact with users, process their inputs, and provide relevant responses. It includes features for handling documents, images, and schematics, with integration with ChromaDB for efficient storage and retrieval. FInally plans are in place to create a Machine learning chat bot type sessions that allows users to interact with their machine learning use cases and understand them better. 

## Project Structure
The chatbot will start, and you can begin interacting with it through the command line interface.
```
llm/
├── app.py                  # Main application entry point
├── config/
│   └── config.yaml        # Configuration settings
│
├── database/              # Database related files
│   ├── app.db
│   └── chroma.sqlite3
│       ├── data_level0.bin
│       ├── header.bin
│       ├── length.bin
│       └── link_lists.bin
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
│   ├── chat.py         # Chat interface
│   ├── data_db.py      # Database interface
│   ├── history.py      # Chat history page
│   └── home.py         # Home page
│
├── src/                 # Source code
│   ├── main.py         # Main logic
│   └── utils.py        # Utility functions
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

3. Install the required dependencies: `` pip install -r requirements.txt``


## Usage

To run the chatbot application, execute the following command from the project root: ``python src/main.py``
## Features

- Interactive chatbot using LLMs mostly using Ollama (llama3.2 : 3 billion model)
- Document processing and analysis
- Image and schematic handling
- Integration with ChromaDB for efficient storage
- Agents that does the same things as RAG
- Further addition of machine learning and chat bot mix

## Dependencies

- Python        3.7+
- Streamlit     1.41.0
- Ollama        0.5.1
- Langchain     0.3.9 

For a complete list of dependencies, refer to the `requirements.txt` file.

## Configuration

[Explain any configuration files or environment variables that need to be set up]

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

## Improvements

1. Right now the pdf can only be added one by one and can only render one pdf at the time. Tried to use st.tabs to render the pdf per tabs but only the first tab is rendered. 
2. 