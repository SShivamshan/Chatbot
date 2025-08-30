import sys
sys.path.append(".")
import argparse
import os
import subprocess
from rich.console import Console
from models.Model import Chatbot,CHATOpenAI
from models.SupervisorAgent import SupervisorAgent
from src.utils import pretty_print_answer,pretty_print_query

def check_ollama_model(model_name):
    """
    Check if a model is available in Ollama
    """
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            available_models = result.stdout
            return model_name in available_models
        else:
            print(f"Warning: Could not check Ollama models: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Warning: Ollama command not found. Make sure Ollama is installed and in your PATH.")
        return False

def get_openai_key():
    """
    Retrieve OpenAI API key from environment or prompt user
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("Error: OpenAI API key is required when using OpenAI models.")
            sys.exit(1)
    return api_key

def validate_args(args):
    """
    Validate the provided arguments
    """
    # Check OpenAI key if using OpenAI
    if args.model == 'openai':
        args.openai_key = get_openai_key()
    
    # Check if Llama model exists in Ollama
    if args.model == 'llama':
        if not check_ollama_model(args.model_type):
            response = input(f"Model '{args.model_type}' not found in Ollama. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                sys.exit(1)

def retrieve_llm(args):
    try:
        provider = args.model
        model_name = args.model_type
        temperature = args.temperature
        max_tokens = args.context_length

        if provider == "openai":
            api_key = args.openai_key
            if not api_key:
                print("OpenAI API key not found in session state", "ERROR")
                return None
                
            handler = CHATOpenAI(
                api_key=api_key,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif provider == "llama":
            handler = Chatbot(
                base_url=args.host,
                model=model_name,
                context_length=max_tokens
            )
        else:
            print(f"Unknown provider: {provider}", "ERROR")
            print("Exiting...")
            sys.exit(1)
                
        return handler
        
    except Exception as e:
        print(f"Can't retrieve llm : {e}")

def create_parser():
    """
    Create and configure the argument parser
    """
    parser = argparse.ArgumentParser(
        description="Chatbot/Agent Interface with configurable models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
python main.py --model llama --model-type llama2 --task 0
python main.py --model openai --model-type gpt-4 --task 1 --temperature 0.5
python main.py --host http://localhost:8080 --context-length 4096
python main.py  # Uses defaults: llama, llama3.2, task 0
        """
    )
    
    parser.add_argument(
        '--model', 
        choices=['llama', 'openai'],
        default='llama',
        help='Model provider to use (default: llama)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='llama3.2:3b',
        help='Specific model type/name (default: llama3.2:3b for llama, gpt-4.1 for openai)'
    )
    
    parser.add_argument(
        '--task',
        type=int,
        choices=[0, 1],
        default=0,
        help='Task mode: 0 for chatbot, 1 for supervisor agent (default: 0)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='http://localhost:11434',
        help='Ollama server URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Model temperature for response randomness (default: 0.7)'
    )
    
    parser.add_argument(
        '--context_length',
        type=int,
        default=8192,
        help='Maximum context length for the model (default: 8192)'
    )
    
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.model == 'llama':
        args.model_type = 'llama3.2:3b'
    elif args.model == 'openai':
        args.model_type = 'gpt-4.1'
    
    validate_args(args)

    print(f"Configuration:")
    print(f"  Model Provider: {args.model}")
    print(f"  Model Type: {args.model_type}")
    print(f"  Task Mode: {'Chatbot' if args.task == 0 else 'Supervisor Agent'}")
    if args.model == 'llama':
        print(f"  Ollama Host: {args.host}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Context Length: {args.context_length}")
    print("-" * 50)
    console = Console()

    if args.task == 0:
        llm = retrieve_llm(args)

        while True:
            query = input("You: ")
            if query.lower() in ['exit', 'quit', 'bye']:
                console.print("[bold red]Exiting chat...[/bold red]")
                break
            try:
                pretty_print_query(query,console)
                answer = llm.invoke(input=query).content
                pretty_print_answer(answer,console)
            except Exception as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                console.print("\n[bold red]Chat interrupted by user![/bold red]")
                break
                
    else:
        llm = retrieve_llm(args)
        agent = SupervisorAgent(chatbot=llm)

        while True:
            query = input("Task: ")
            if query.lower() in ['exit', 'quit', 'bye']:
                console.print("[bold red]Exiting Agent chat... [/bold red]")
                break
            try:
                pretty_print_query(query,console)
                result = agent.run(query,filename=None)
                final_answer = ""
                for response in list(result.values()):
                    answer_text = response["final_answer"]
                    # Add to final_answer with proper spacing
                    if final_answer:  # If final_answer already has content, add a separator
                        final_answer += " ".join(answer_text)
                    else:
                        final_answer = answer_text
                pretty_print_answer(final_answer,console=console)
            except Exception as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                console.print("\n[bold red]Chat interrupted by user![/bold red]")
                break

if __name__ == "__main__":
    main()
    