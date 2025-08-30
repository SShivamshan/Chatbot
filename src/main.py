import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
#       Finish documentation 
#       Create a script for the ollama server
#       Add DOCKER perhaps
#       The torch cuda is not working find a solution for that and find a way to see it the model is already in the GPU for when we read the pdf since it' uses the yolo models 

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
