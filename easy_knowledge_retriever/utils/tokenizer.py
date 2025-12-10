import abc
import asyncio
import logging
import os
import tiktoken
from .logger import logger

class TokenizerInterface(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass


class Tokenizer(TokenizerInterface):
    def __init__(self, tokenizer_model: str):
        self.tokenizer_model = tokenizer_model

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


class TiktokenTokenizer(TokenizerInterface):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(
                f"Model {model_name} not found in tiktoken. Falling back to cl100k_base."
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)


def truncate_list_by_token_size(
    list_data: list, key: callable, max_token_size: int
) -> list:
    """Truncate the list of data by token size.

    Args:
        list_data: The list of data to truncate.
        key: The key function to get the text from the data.
        max_token_size: The maximum token size.

    Returns:
        The truncated list of data.
    """
    if max_token_size <= 0:
        return []

    # Get the embedding binding to use its tokenizer
    # We use embedding binding because content truncation is usually for context window
    # get_llm_binding and llm.binding_options are currently unavailable.
    # We will rely on the default Tokenizer logic below.
    embedding_binding = None

    if embedding_binding is None:
        # Fallback to rough estimation if no binding available
        current_token_size = 0
        truncated_list = []
        for item in list_data:
            text = key(item)
            # Rough estimation: 1 token ~ 4 characters
            token_size = len(text) / 4
            if current_token_size + token_size > max_token_size:
                break
            truncated_list.append(item)
            current_token_size += token_size
        return truncated_list

    # Use binding's tokenizer if available (needs implementation in bindings first)
    # For now, use TiktokenTokenizer as a robust default for most models
    tokenizer = TiktokenTokenizer(
        os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    current_token_size = 0
    truncated_list = []

    for item in list_data:
        text = key(item)
        encoded_tokens = tokenizer.encode(text)
        token_size = len(encoded_tokens)

        if current_token_size + token_size > max_token_size:
            break

        truncated_list.append(item)
        current_token_size += token_size

    return truncated_list
