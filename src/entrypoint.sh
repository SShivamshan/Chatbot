#!/bin/bash

# Mostly through these : https://stackoverflow.com/questions/78522325/running-ollama-in-docker-file-for-python-langchain-application 
# https://stackoverflow.com/questions/78232178/ollama-in-docker-pulls-models-via-interactive-shell-but-not-via-run-command-in-t 
# Used their bash script for our use case and thanks to the saving of the pid of the server we can just start teh server and end it with ctrl+c

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed."
    exit 1
fi

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &

# Save the PID of the server
SERVER_PID=$!

# Give server a few seconds to start
sleep 5

# Check models
models=("llava:7b" "llama3.2:3b" "nomic-embed-text:latest")
for model in "${models[@]}"; do
    if ollama list | grep -q "$model"; then
        echo "Model $model is available."
   else
        echo "Model $model is NOT pulled. Pulling now..."
        ollama pull "$model"
        if [ $? -eq 0 ]; then
            echo "Successfully pulled $model."
        else
            echo "Failed to pull $model."
        fi
    fi
done

echo "Server is running. Press Ctrl+C to stop."

# Wait for the server process to finish
wait $SERVER_PID