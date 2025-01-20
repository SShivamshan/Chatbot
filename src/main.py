import sys
sys.path.append(".")
from app import ChatbotApp
from models.RAG import RAG
from models.Model import Chatbot
from utils import Vectordb
# TODO: 
#       RAG works with chat history but needs some works perhaps using unstructured should be allowed to view how it works  
#       Procced with unstructered module to get the data and images and save images and tables in chromadb 
#       Correct the agent 
#       Thursday : save everything to github


if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
    