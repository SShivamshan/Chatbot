import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
# Now we need add the entier process to the app start with Webagent then go to the rest and also need to rich.console.Console(record=True) and console.export_html() these to render on streamlit
# Test entier process altogether
# Add The llm handler to the app and test 
# Add teh method to use properly the agent through cmd using run_agent.py 

#       The torch cuda is not working find a solution for that and find a way to see it the model is already in the GPU. 
#       Finish documentation 
#       Create a script for the ollama server
#       Create a method to ensure security for the app.db

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
