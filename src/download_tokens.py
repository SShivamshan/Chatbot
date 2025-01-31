import argparse
import nltk
from nltk.data import find
from typing import List

def download_nltk_tokernizers(tokens:List[str]):
    """
    Downloads NLTK tokenizers if not already available in the system.
    Args:
        tokens (List[str]): List of NLTK tokenizers to check and download.
    Returns:
        None
    """
    for token in tokens:
        try:
            # Check if the tokenizer is already installed
            find(f"tokenizers/{token}")
            print(f"Tokenizer '{token}' is already available.")
        except LookupError:
            # If not installed, download the tokenizer
            print(f"Downloading tokenizer: {token}")
            nltk.download(token)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--download-nltk-tokernizers", nargs='+', default=["punk_tab", "stopwords", "averaged_perceptron_tagger_eng"],
                        help="List of NLTK tokenizers to download")
    
    args = parser.parse_args()

    
    download_nltk_tokernizers(tokens=args.download_nltk_tokernizers)
