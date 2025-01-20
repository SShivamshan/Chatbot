from langchain.llms.base import LLM
import requests
from typing import Optional, List
from pydantic import Field
import json


class Chatbot(LLM):
    base_url: str = Field(None,alias='base_url')
    model: str = Field(None,alias='model')
    context_length: int = Field(None,alias='context_length')

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b", context_length: int = 18000):
        """
        A chatbot class leveraging Ollama's local LLM.

        Args:
            base_url (str): The local Ollama server URL.
            model (str): The name of the model to use (e.g., 'llama3.2:3b').
            context_length (int): Maximum context length for the chatbot set to 18000 for performance reasons on personnal computer.
        """
        super(Chatbot,self).__init__()
        self.base_url = base_url
        self.model = model
        self.context_length = context_length

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Make a call to the local Ollama API.

        Args:
            prompt (str): The prompt to send to the LLM.
            stop (Optional[List[str]]): Optional stop tokens.

        Returns:
            str: The generated response from the model.
        """
        payload = {
        "model": self.model,
        "prompt": prompt,
        "context_length": self.context_length
        }
        if stop:
            payload["stop"] = stop
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True)
            response.raise_for_status()

            # print("Raw response content:", response.text)
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
                        print(f"Skipping invalid JSON line: {line}")  # Debugging purposes

            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with Ollama server: {e}")

        except ValueError as e:
            raise RuntimeError(f"Invalid response from the API: {e}")

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
        Return the type of LLM being used. # Thinking of adding OpenAI

        Returns:
            str: The LLM type ("ollama").
        """
        return "ollama"


