import sys
sys.path.append(".")
import logging
from models.Model import Chatbot
from models.WebAgent import WebAgent
from src.utils import pretty_print_answer

def main():
    chatbot = Chatbot()
    agent = WebAgent(chatbot=chatbot,
              log_level=logging.INFO,    
              pretty_print=True )
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        answer = agent.run(query)
        pretty_print_answer(answer)



if __name__ == "__main__":
    main()

    