import sys
sys.path.append(".")
from app import ChatbotApp
# TODO: 
#       Add DOCKER perhaps

if __name__ == "__main__": 
    app = ChatbotApp()
    app.run()
