import sys
sys.path.append(".")
from app import ChatbotApp

# TODO: 
#       Find a way so that we delete an instance of saved mode in case the user adds a new pdf 
#       Correct the agent and add it as the third componen
#       The torch cuda is not working find a solution for that 
#       Finish settings and config.toml
#       Finish documentation 
#       Create a script for the ollama server
#       Create a method to ensure security for the app.db
#       Look into the message : 2025-01-27 17:14:07.968 Examining the path of torch.classes raised: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()

