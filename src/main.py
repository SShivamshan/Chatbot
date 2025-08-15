import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
# Finish PDF and COde agent integration test and followed by agentic rag and superviosor agent and then prepare a method to show sources appropriately based on the answers
# Add The llm handler to the app and test 
# Add teh method to use properly the agent through cmd using run_agent.py 

#       The torch cuda is not working find a solution for that and find a way to see it the model is already in the GPU. 
#       Finish documentation 
#       Create a script for the ollama server
#       Create a method to ensure security for the app.db

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
