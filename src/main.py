import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
# Only need to add user_id upon ImageMangaer and Table Manager. Only need to finish the account management, deleting and admin process. 
# Need to add popover messages. 
# Add the method to use properly the agent through cmd using run_agent.py 

#       The torch cuda is not working find a solution for that and find a way to see it the model is already in the GPU for when we read the pdf since it' uses the yolo models 
#       Finish documentation 
#       Create a script for the ollama server
#       Create a method to ensure security for the app.db

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
