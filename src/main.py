import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
#       unexpected '{' in field name
#       when the we send another message it seems that the chat history gets nothing. 
#       Correct the agent and add it as the third component
#       Create a script for the ollama server
#       Create a method to ensure security for the app.db
#       Look into the message : 2025-01-27 17:14:07.968 Examining the path of torch.classes raised: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()

    