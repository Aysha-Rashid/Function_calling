from typing import Any, Dict, Optional
from pydantic import BaseModel as base
from enum import Enum
from dataclasses import dataclass, field

class Paraschema(base):
    type: str

class FunctionDefinition(base):
    name: str
    description: str
    parameters: Dict[str, Paraschema]
    returns: Dict[str, Any]


class Prompts(base):
    prompt: str


class CurrentState(Enum):
    FUNCTION_NAME = "Function name state"
    AFTER_FUNCTION_NAME = "After function state"
    PARAMETER_NAME = "Parameter name state"
    AFTER_PARAMETER_NAME = "After parameter state"
    PARAMETER_KEY = "Parameter key state"
    PARAMETER_VALUE = "Paramter value state"
    END = "Needs to end the generation"


@dataclass
class DecodingStateJson():
    phase: CurrentState = CurrentState.FUNCTION_NAME
    selected_function_name: Optional[str] = None
    parameter_index: int = 0
    generated_text: str = ""
    Functions_list: list[FunctionDefinition] = field(default_factory=list)
    valid_function_name: dict[str, FunctionDefinition] = field(default_factory=dict, init=False)
    selected_function: str = ""
    parameter_key: dict[str, str] = field(default_factory=dict, init=False)
    parameter_value: dict[Any, Any] = field(default_factory=dict, init=False)
    def __post_init__(self):
        self.valid_function_name = {func.name: func for func in self.Functions_list}


#this is after the whole generation (to validate whether the format is right)
def validate_function_call(call: dict[str, Any], functions: list[FunctionDefinition]) -> None:
    """

    """
    expected_keys = {"name", "parameters"}
    if set(call) != expected_keys:
        raise ValueError("Function call must contain exactly")

    function_name = call["name"]
    if not isinstance(function_name, str):
        raise ValueError("'name' must be a string")
    selected_function = next(
        (function for function in functions if function.name == function_name),
        None,
    )
    if selected_function is None:
        raise ValueError(f"Unknown function: {function_name}")
    parameters = call["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("'arguments' must be a JSON object")
    expected_para = set(selected_function.parameters)
    received_arguments = set(parameters)
    missing_arguments = expected_para - received_arguments
    if missing_arguments:
        raise ValueError(f"Missing arguments for {function_name}: {sorted(missing_arguments)}")
    extra_arg = received_arguments - expected_para
    if extra_arg:
        raise ValueError(f"Unexpected arguments for {function_name}: {sorted(missing_arguments)}")
    for parameter_name, parameter in selected_function.parameters.items():
        value = parameters[parameter_name]
        if parameter.type == "string" and not isinstance(value, str):
            raise ValueError(f"{parameter_name} must be a string")
        if parameter.type == "number":
            is_number = (isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    )
            if not is_number:
                raise ValueError(f"'{parameter_name}' must be a finite number")
        if parameter.type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"'{parameter_name}' must be a boolen")
    return selected_function    


class FunctionCallGrammar:
    def __init__(self, functions: list[FunctionDefinition]) -> None:
        self.functions_by_name = {
            function.name: function
            for function in functions
        }
    # def is_valid_prefix(self, text: str) -> bool:

    def is_complete(self, text: str) -> bool:
        try:
            call = json.loads(text)
        except json.JSONDecodeError:
            return False
        try:
            validate_function_call(call, self.functions_by_name.values())
        except ValueError:
            return False
        return True

    
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
    
class FunctionCall(base):
    name: str
    parameters: Dict[str, Any]


def function_to_call_schema(function: FunctionDefinition) -> Dict[str, Any]:
    parameters_properties = {
        parameter_name: {"type": parameter.type}
        for parameter_name, parameter in function.parameters.items()
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"const": function.name},
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": parameters_properties,
                "required": list(parameters_properties),
            },
        },
        "required": ["name", "parameters"],
    }


def functions_to_call_schema(functions: list[FunctionDefinition]) -> Dict[str, Any]:
    return {
        "oneOf": [
            function_to_call_schema(function)
            for function in functions
        ]
    }