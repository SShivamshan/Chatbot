import logging
import json
import time
from typing import Any, Dict, Optional
from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.text import Text
# SOLUTION adapated from : https://medium.com/@ahmedharabi/get-started-with-the-python-rich-library-2736b1b57941 

class AgentLogger:
    """
    A logging utility for tracking and displaying Agent workflow steps and state.
    """
    
    def __init__(self, log_level=logging.INFO, console_output=True, file_output=False, 
                 log_file="agent_logs.log", pretty_print=True):
        """
        Initialize the logger with desired output formats.
        
        params:
            log_level: The minimum logging level to record
            console_output: Whether to output logs to console
            file_output: Whether to output logs to a file
            log_file: Name of the log file if file_output is True
            pretty_print: Whether to use Rich for formatted console output
        """
        self.log_level = log_level
        self.console_output = console_output
        self.file_output = file_output
        self.log_file = log_file
        self.pretty_print = pretty_print
        
        # Initialize logger
        self.logger = logging.getLogger("AgentLogger")
        self.logger.setLevel(self.log_level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Console handler with Rich formatting if enabled
        if self.console_output:
            if self.pretty_print:
                self.console = Console()
                console_handler = RichHandler(rich_tracebacks=True, console=self.console)
            else:
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
            
            console_handler.setLevel(self.log_level)
            self.logger.addHandler(console_handler)
        
        # File handler if enabled
        if self.file_output:
            file_handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.log_level)
            self.logger.addHandler(file_handler)
        
        # Metrics tracking
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "total_time": None,
            "node_times": {},
            "node_sequence": [],
            "steps_completed": 0,
            "node_count": {}
        }
    
    def format_state(self, state: Dict[str, Any], truncate_length: int = 500) -> str:
        """Format GraphState for better readability."""
        formatted_state = {}
        
        for key, value in state.items():
            if isinstance(value, str) and len(value) > truncate_length:
                formatted_state[key] = f"{value[:truncate_length]}... [truncated, {len(value)} chars total]"
            elif isinstance(value, list) and len(str(value)) > truncate_length:
                if len(value) > 0 and isinstance(value[0], dict):
                    formatted_state[key] = f"[{len(value)} items] - First: {str(value[0])[:100]}..."
                else:
                    formatted_state[key] = f"[List with {len(value)} items, truncated]"
            elif isinstance(value, dict) and len(str(value)) > truncate_length:
                formatted_state[key] = f"[Dict with {len(value)} keys, truncated]"
            else:
                formatted_state[key] = value
                
        return formatted_state
    
    def start_agent_run(self, query: str):
        """Log the start of an agent run with the given query."""
        self.metrics["start_time"] = time.time()
        self.metrics["node_sequence"] = []
        self.metrics["steps_completed"] = 0
        self.metrics["node_count"] = {}
        
        if self.pretty_print and self.console_output:
            self.console.print(Panel(f"[bold red]Starting agent run[/bold red]\n\n[yellow]Query:[/yellow] {query}", 
                                     title="Agent Run Started", border_style="red"))
        else:
            self.logger.info(f"Starting agent run with query: {query}")
    
    def log_node_entry(self, node_name: str, state: Dict[str, Any]):
        """Log the entry into a workflow node."""
        self.metrics["node_sequence"].append(node_name)
        self.metrics["node_count"][node_name] = self.metrics["node_count"].get(node_name, 0) + 1
        node_start_time = time.time()
        
        if self.pretty_print and self.console_output:
            formatted_state = self.format_state(state)
            
            # Create a table for the state
            table = Table(title=f"Current State", show_header=True)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="yellow")
            
            for key, value in formatted_state.items():
                str_value = str(value)
                if len(str_value) > 80:  # Further truncate for table display
                    str_value = str_value[:77] + "..."
                table.add_row(key, str_value)
            
            self.console.print(Panel(f"[bold green]Entering node:[/bold green] [bold cyan]{node_name}[/bold cyan]", 
                                     title=f"Node {self.metrics['steps_completed'] + 1}", border_style="green"))
            self.console.print(table)
        else:
            self.logger.info(f"Entering node: {node_name}")
            self.logger.debug(f"Current state: {json.dumps(self.format_state(state), indent=2)}")
        
        return node_start_time
    
    def log_node_exit(self, node_name: str, state: Dict[str, Any], node_start_time: float):
        """Log the exit from a workflow node."""
        node_time = time.time() - node_start_time
        self.metrics["node_times"][node_name] = self.metrics["node_times"].get(node_name, 0) + node_time
        self.metrics["steps_completed"] += 1
        
        if self.pretty_print and self.console_output:
            formatted_state = self.format_state(state)
            
            # Create a table for updated state
            table = Table(title=f"Updated State", show_header=True)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="yellow")
            
            for key, value in formatted_state.items():
                str_value = str(value)
                if len(str_value) > 80:
                    str_value = str_value[:77] + "..."
                table.add_row(key, str_value)
            
            self.console.print(Panel(
                f"[bold red]Exiting node:[/bold red] [bold cyan]{node_name}[/bold cyan]\n"
                f"[bold]Time taken:[/bold] {node_time:.2f}s", 
                title=f"Node {self.metrics['steps_completed']} Completed", 
                border_style="red"
            ))
            self.console.print(table)
            self.console.print("\n")
        else:
            self.logger.info(f"Exiting node: {node_name} (Time: {node_time:.2f}s)")
            self.logger.debug(f"Updated state: {json.dumps(self.format_state(state), indent=2)}")
    
    def log_decision(self, node_name: str, next_nodes: list):
        """Log a routing decision in the workflow."""
        if self.pretty_print and self.console_output:
            self.console.print(f"[bold magenta]Decision:[/bold magenta] From [cyan]{node_name}[/cyan] → "
                               f"To {', '.join([f'[green]{n}[/green]' for n in next_nodes])}")
        else:
            self.logger.info(f"Routing decision: From {node_name} to {', '.join(next_nodes)}")
    
    def log_error(self, node_name: str, error: Exception, state: Optional[Dict[str, Any]] = None):
        """Log an error that occurred during execution."""
        if self.pretty_print and self.console_output:
            error_text = Text.from_markup(f"[bold red]Error in {node_name}:[/bold red] {str(error)}")
            self.console.print(Panel(error_text, title="Error Occurred", border_style="red"))
            if state:
                self.console.print("State at error:")
                self.console.print(json.dumps(self.format_state(state), indent=2))
        else:
            self.logger.error(f"Error in {node_name}: {str(error)}")
            if state:
                self.logger.debug(f"State at error: {json.dumps(self.format_state(state), indent=2)}")
                
    def log_tool_call(self, tool_name: str, inputs: Dict[str, Any], outputs: Any = None):
        """Log a tool call with inputs and outputs."""
        if self.pretty_print and self.console_output:
            # Create a table for the tool call
            table = Table(title=f"Tool Call: {tool_name}", show_header=True)
            table.add_column("Parameter", style="bold cyan")
            table.add_column("Value", style="yellow")
            
            for key, value in inputs.items():
                str_value = str(value)
                if len(str_value) > 80:
                    str_value = str_value[:77] + "..."
                table.add_row(key, str_value)
            
            self.console.print(table)
            
            if outputs is not None:
                if isinstance(outputs, dict):
                    # Create a table for the outputs
                    output_table = Table(title="Tool Output", show_header=True)
                    output_table.add_column("Key", style="bold green")
                    output_table.add_column("Value", style="yellow")
                    
                    for key, value in outputs.items():
                        str_value = str(value)
                        if len(str_value) > 80:
                            str_value = str_value[:77] + "..."
                        output_table.add_row(key, str_value)
                    
                    self.console.print(output_table)
                else:
                    trunc_output = str(outputs)
                    if len(trunc_output) > 500:
                        trunc_output = trunc_output[:500] + "... [truncated]"
                    self.console.print(f"[bold green]Tool Output:[/bold green]\n{trunc_output}")
        else:
            self.logger.info(f"Tool Call: {tool_name}")
            self.logger.debug(f"Tool Inputs: {json.dumps(inputs, indent=2)}")
            if outputs is not None:
                if isinstance(outputs, dict):
                    self.logger.debug(f"Tool Outputs: {json.dumps(outputs, indent=2)}")
                else:
                    self.logger.debug(f"Tool Output: {outputs}")
    
    def end_agent_run(self, final_result: Dict[str, Any]):
        """Log the completion of an agent run with results."""
        self.metrics["end_time"] = time.time()
        self.metrics["total_time"] = self.metrics["end_time"] - self.metrics["start_time"]
        
        if self.pretty_print and self.console_output:
            # Create a metrics table
            metrics_table = Table(title="Run Metrics", show_header=True)
            metrics_table.add_column("Metric", style="bold cyan")
            metrics_table.add_column("Value", style="yellow")
            
            metrics_table.add_row("Total Time", f"{self.metrics['total_time']:.2f}s")
            metrics_table.add_row("Steps Completed", str(self.metrics["steps_completed"]))
            metrics_table.add_row("Node Sequence", " → ".join(self.metrics["node_sequence"]))
            
            for node, count in self.metrics["node_count"].items():
                metrics_table.add_row(f"Node {node} count", str(count))
                
            for node, time_taken in self.metrics["node_times"].items():
                metrics_table.add_row(f"Node {node} time", f"{time_taken:.2f}s")
            
           # Extract the answer for display
            answer = final_result.get("final_answer", "No answer generated.")
            if answer == "No answer generated.":
                display_answer = answer
            elif isinstance(answer, str):
                # If it's already a string, use it directly
                display_answer = answer
            else:
                # If it's an AI message object, extract the content
                display_answer = answer.content

            # Truncate if too long
            if len(display_answer) > 500:
                display_answer = display_answer[:500] + "... [truncated]"

            self.console.print(Panel(
                f"[bold red]Agent run completed[/bold red]\n\n"
                f"[yellow]Final Answer:[/yellow]\n{display_answer}",
                title="Run Completed",
                border_style="red"
            ))
            self.console.print(metrics_table)
        else:
            self.logger.info(f"Agent run completed in {self.metrics['total_time']:.2f}s")
            self.logger.info(f"Steps completed: {self.metrics['steps_completed']}")
            self.logger.info(f"Node sequence: {' → '.join(self.metrics['node_sequence'])}")
            self.logger.info(f"Final answer: {final_result.get('final_answer', 'No answer generated.')}")

# Decorator for instrumenting Agent class methods
def log_method(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            method_name = func.__name__
            logger.logger.debug(f"Calling method: {method_name}")
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                logger.logger.debug(f"Method {method_name} completed in {time.time() - start_time:.2f}s")
                return result
            except Exception as e:
                logger.log_error(method_name, e)
                raise
        return wrapper
    return decorator