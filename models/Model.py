from langchain.llms.base import LLM
import requests
from typing import Optional, List, Any
from pydantic import Field, BaseModel
import json
import logging
import subprocess
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI

class Chatbot(ChatOllama):
    base_url: str = Field(None,alias='base_url')
    model: str = Field(None,alias='model')
    context_length: int = Field(None,alias='context_length')
    logger: logging.Logger = Field(None, alias='logger')

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b", context_length: int = 18000 ):
        """
        A chatbot class leveraging Ollama's local LLMs (llama3.2:3b, )

        Args:
            base_url (str): The local Ollama server URL.
            model (str): The name of the model to use (e.g., 'llama3.2:3b').
            context_length (int): Maximum context length for the chatbot set to 18000 for performance reasons on personnal computer.
        """
        super(Chatbot,self).__init__()
        self.base_url = base_url
        self.model = model
        self.context_length = context_length
        self.logger = self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)  # Local variable
        return logger

    def _call(self, prompt: str, image: Optional[str] = None, stop: Optional[List[str]] = None) -> str: # type: ignore
        """
        Make a call to the local Ollama API.

        Args:
            prompt (str): The prompt to send to the LLM.
            image (Optional[str]): A base64-encoded image string (optional).
            stop (Optional[List[str]]): Optional stop tokens.

        Returns:
            str: The generated response from the model.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "context_length": self.context_length,
            "temperature" : 0.1                     # Solution : https://github.com/ollama/ollama/issues/6410 
        }

        if stop:
            payload["stop"] = stop

        # Directly include the base64-encoded image
        if image and self.model.startswith("llava"):
            payload["image"] = image

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True)
            response.raise_for_status()

            # Process the streamed response
            generated_text = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():  # Ignore empty lines
                    try:
                        data = json.loads(line)  # Parse each JSON object
                        generated_text += data.get("response", "")  # Append "response" field
                        if data.get("done"):  # Stop if "done" is true
                            break
                    except json.JSONDecodeError:
                        self.logger.info(f"Skipping invalid JSON line: {line}")  # Debugging purposes

            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with Ollama server: {e}")

        except ValueError as e:
            raise RuntimeError(f"Invalid response from the API: {e}")


    def unload_model(self, model_name:str=None): # Solution : https://github.com/ollama/ollama/issues/1600 
        """
        Unload the current model or the given model from memory by calling the Ollama API.
        """
        model_to_unload = model_name if model_name else self.model

        if not model_to_unload:  # Ensure a model name is available
            self.logger.error("No model specified to unload.")
            return

        curl_command = [
            "curl",
            f"{self.base_url}/api/generate",
            "-d", f'{{"model": "{model_to_unload}", "keep_alive": 0}}'
        ]

        try:
            result = subprocess.run(curl_command, capture_output=True, text=True, check=True)

            # Check if the command was successful
            if result.returncode == 0:
                self.logger.info(f"Successfully unloaded model: {self.model}")
            else:
                self.logger.info(f"Failed to unload model: {result.stderr}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error unloading model: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")

    @property
    def _identifying_params(self) -> dict:
        """
        Return parameters that uniquely identify this LLM.

        Returns:
            dict: Model-specific parameters.
        """
        return {
            "model": self.model,
            "context_length": self.context_length
        }

    @property
    def _llm_type(self) -> str:
        """
        Return the type of LLM being used. Such as llama3.2 would be only for text based answers and 
        llava3.2 would be for image-based summaries.  

        Returns:
            str: The LLM type ("ollama" or "llava").
        """
        return self.model
    
    @property
    def llm_properties(self) -> dict:
        """
        Return the properties of the LLM.

        Returns:
            dict: Properties of the LLM.
        """
        return {
            "type": self._llm_type,
            "identifying_params": self._identifying_params
        }


class CHATOpenAI(ChatOpenAI):
    """Handles interactions with the LLM for code translation using OpenAI API"""
    api_key: str = Field(None,alias='api_key')
    model: str = Field(default="gpt-4.1",alias="model")
    max_tokens: int = Field(default=8192,alias="max_tokens")
    temperature: float = Field(default=0.0,alias="temperature")
    client:Optional[Any] = Field(default=None, exclude=True)
    logger: Optional[Any] = Field(default=None, exclude=True)

    def __init__(self, api_key=None, 
                 model="gpt-4.1", 
                 max_tokens=8192,
                 temperature=0.0):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key 
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.logger = self._setup_logging()
        self.logger.info(f"LLMHandler initialized with model {self.model}")
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _call(self, prompt: str, image=None):
        """
        Sends a prompt or an image (with optional prompt) to the OpenAI API.

        Args:
            prompt (str): The textual prompt to send.
            image (optional): The image input. Could be a path, bytes, or a special object. Default is None.

        Returns:
            str or dict: The API response content, string for text completions or dict for image.
        """
        try:
            if image is None:
                self.logger.info("Sending text prompt to OpenAI API.")
                response = self.invoke([{"role": "user", "content": prompt}])
                self.logger.info("Received text completion response.")
                return response.content
            else:
                self.logger.info("Sending image with prompt to OpenAI API.")
            
                messages = [
                    {"role": "system", "content": "You are an image understanding assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image}}
                    ]}
                ]
                response = self.invoke(messages)
                self.logger.info("Received image understanding response.")
                return response.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            raise

    @property
    def _llm_type(self) -> str:
        """
        Return the type of LLM being used. 

        Returns:
            str: The LLM type .
        """
        return self.model

    
