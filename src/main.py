import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
#       Now we Need to find some improvements needed for datetime 
#       and finalize the code agent
#       End with  web search and web scrapping together comes into place and also the cases of multiple
#       The torch cuda is not working find a solution for that and find a way to see it the model is already in the GPU. 
#       Finish settings and config.toml -> do it every friday
#       Finish documentation 
#       Create a script for the ollama server
#       Create a method to ensure security for the app.db
#       Look into the message : 2025-01-27 17:14:07.968 Examining the path of torch.classes raised: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
