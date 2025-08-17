import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
# Add The llm handler to the app and test 
# Add the method to use properly the agent through cmd using run_agent.py 

#       The torch cuda is not working find a solution for that and find a way to see it the model is already in the GPU for when we read the pdf since it' uses the yolo models 
#       Finish documentation 
#       Create a script for the ollama server
#       Create a method to ensure security for the app.db

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
