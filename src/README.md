## SRC Folder

The src folder contains all the core components required to run the project. It includes utility functions, classes, and scripts that enable both chatbot and agent functionalities.

**Key Components**

- `run_agent.py`: Launches a chatbot or an agent from the command line interface.
- `download_tokens.py`: Downloads all necessary NLTK tokens required for natural language processing.
- `AgentLogger.py`: Provides logging functionality to track the agent's actions.
- `entrypoint.sh`: Starts the Ollama server and verifies the presence of default models.
- `utils.py`: Contains utility functions and helper methods used across the project.
- `main.py`: Serves as an entry point for running the application programmatically.

## Structure

```
src
├── AgentLogger.py          # Provides logging functionality to track the agent's actions.
├── download_tokens.py      # Downloads all necessary NLTK tokens required for natural language processing.
├── entrypoint.sh           # Starts the Ollama server and verifies the presence of default models.
├── main.py                 # Serves as an entry point for running the application programmatically.
├── README.md               
├── run_agent.py            # Launches a chatbot or an agent from the command line interface.
└── utils.py                # Contains utility functions and helper methods used across the project.
```

## Run agent through command line

The run_agent.py script provides a command-line interface to interact with either a chatbot or a supervisor agent, using configurable models from Ollama (llama) or OpenAI (openai). It supports advanced features such as querying local files, syntax-highlighted code outputs, and agent task execution.

### Key Features

- Run a chatbot or a supervisor agent interactively.
- Supports llama models (via Ollama) and openai models.
- Optional temperature and context length settings for LLM responses which allows the control over GPU usage. 
- Detects and pretty-prints code and query outputs.
- Allows /file flag to include local files for agent tasks.
- Handles graceful exit, clearing the screen, and keyboard interrupts.

### Usage

```
# Run the default chatbot using Ollama's llama3.2:3b model
python run_agent.py

# Run a supervisor agent with Ollama
python run_agent.py --model llama --model-type llama3.2:3b --task 1

# Run OpenAI GPT-4.1 model with a custom temperature
python run_agent.py --model openai --model-type gpt-4.1 --temperature 0.5

# Connect to a custom Ollama host and specify context length
python run_agent.py --host http://localhost:8080 --context-length 4096
```

## Entrypoint.sh

The entrypoint.sh script is used to start the Ollama server and ensure that the required models are available locally. It automates the setup process for running LLMs with the project.

### Key Features

- Checks if Ollama is installed on the system.
- Starts the Ollama server in the background.
- Verifies that required models (llava:7b, llama3.2:3b, nomic-embed-text:latest) are available.
- Automatically pulls any missing models.
- Keeps the server running until manually stopped (Ctrl+C).

### Usage 
```
# Make sure the script is executable
chmod +x entrypoint.sh

# Start the server and pull required models
./entrypoint.sh
```

### Fixing “port 11434 already in use” error

If you encounter an error saying that port 11434 is already bound or in use, it usually means another Ollama process (or another service) is occupying it.

1. Stop Ollama service (if running via systemd): `sudo systemctl stop ollama`
2. Check what process is using the port: `lsof -i :11434` or `sudo netstat -tulnp | grep 11434`
3. Kill the process manually (replace **PID** with the actual process ID from step 2): `kill -9 <PID>`
4. Restart your application, the port should now be free.

## Agent Logger

The AgentLogger module provides a comprehensive logging utility for tracking and visualizing the workflow of your chatbot or supervisor agent. It leverages the Rich library for visually appealing console output, including tables, panels, and color-coded messages.

### Key Features

- Agent run tracking: Logs the start and end of an agent run, including query inputs and final results.
- Node-level logging: Tracks entry and exit of workflow nodes, including execution time and intermediate state.
- Tool call logging: Logs inputs and outputs of tools used by the agent.
- Decision logging: Tracks routing decisions between workflow nodes.
- Error handling: Logs errors with optional state snapshots.
- Rich formatting: Pretty-prints logs using Rich panels and tables; can also record logs for HTML export.
- Metrics tracking: Collects statistics such as node times, step counts, and total run duration.

