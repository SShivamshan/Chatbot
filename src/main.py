import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
#       TEST WEBscrape tool with another online link
#       Add the pdf file use case with one either a link towards the read pdf or towards the an online pdf 
#       End with cases of multiple and knowledge base
#       when testing within the app need to look into the code aspect for input
#       The torch cuda is not working find a solution for that and find a way to see it the model is already in the GPU. 
#       Finish settings and config.toml -> do it every friday
#       Finish documentation 
#       Create a script for the ollama server
#       Create a method to ensure security for the app.db
#       Look into the message : 2025-01-27 17:14:07.968 Examining the path of torch.classes raised: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
