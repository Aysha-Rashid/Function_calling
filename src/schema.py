from typing import Any, Dict
from pydantic import BaseModel as base
# from dataclass import dataclass, field


class Paraschema(base):
    type: str

class FunctionDefinition(base):
    name: str
    description: str
    parameters: Dict[str, Paraschema]
    returns: Dict[str, Any]


class Prompts(base):
    prompt: str


class DecodingStateJson(base):
    Function_name: str
    After_Function_name: str
    Parameter_name: str
    After_Parameter_name: str
    Parameter_key: str
    Parameter_value: str
    Parameter_partial: str
    End: str


class Tokenizer:
    """
        A tokenizer for rewriting encoder and decorder function using the vocab map
        of the model
    """
    def __init__(self, vocab_map):
        self.vocab_map = vocab_map
        self.token_to_id = {
            token_str: token_id
            for token_str, token_id in self.vocab_map.items()
        }
        self.id_to_token = {
            token_id: token_str
            for token_id, token_str in self.vocab_map.items()
        }


    def encoder(self, input: str) -> list[int]:
        """
            Takes input from the user and searchs the vocab_map to find its token id
        """
        token_id = self.token_to_id.get(input)
        if token_id is None:
            raise ValueError(f"No token ID found for: {input}")
        return [token_id]


    def decoder(self, token_ids: list[int]) -> list[str]:
        result = []
        for token_id in token_ids:
            token_str = self.id_to_token.get(token_id)
            if token_str is None:
                raise ValueError(f"No token found for ID: {token_id}")
            result.append(token_str)
        return result

# @dataclass
# class stateMachine():
    