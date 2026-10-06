from enum import Enum
from typing import Any, Dict, List, Set

from pydantic import BaseModel


class ParameterSchema(BaseModel):
    """A class representing a parameter schema."""
    type: str


class FunctionDefinition(BaseModel):
    """A class representing a function definition."""
    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: Dict[str, Any]


class Prompt(BaseModel):
    """A class representing a prompt for a function call."""
    prompt: str


class Result(BaseModel):
    """A class representing the result of a function call."""
    prompt: str
    name: str
    parameters: Dict[str, Any]


class FunctionCallResult(BaseModel):
    """A class representing the result of a function call."""
    prompt: str
    name: str
    parameters: Dict[str, Any]


class GenerationState(Enum):
    """A class representing the different states of the generation process."""
    FUNCTION_NAME = "FUNCTION_NAME"
    AFTER_FUNCTION_NAME = "AFTER_FUNCTION_NAME"
    PARAMETER_KEY = "PARAMETER_KEY"
    AFTER_PARAMETER_KEY = "AFTER_PARAMETER_KEY"
    PARAMETER_VAL = "PARAMETER_VAL"
    AFTER_PARAMETER_VAL = "AFTER_PARAMETER_VAL"
    END = "END"


class JSONStructure(BaseModel):
    """Runtime state for token-by-token constrained generation."""
    def __init__(
        self,
        functions: List[FunctionDefinition],
        prefix_cache: Dict[tuple[frozenset[str], str], List[int]] | None = None,
    ) -> None:
        self.functions = functions

        self.valid_func_names: Set[str] = {
            function.name for function in functions
        }
        self.param_keys_by_func: Dict[str, Set[str]] = {
            function.name: set(function.parameters.keys())
            for function in functions
        }

        # Shared across prompts because function definitions do not change.
        self.prefix_cache = prefix_cache if prefix_cache is not None else {}

        self.state = GenerationState.FUNCTION_NAME

        self.selected_function: str | None = None
        self.selected_function_object: FunctionDefinition | None = None
        self.remaining_parameters: Set[str] = set()

        self.cur_para: str | None = None
        self.cur_para_type: str | None = None
        self.generated_parameters: Set[str] = set()

        self.partial_para: str = ""
        self.part_para_val: str = ""

        # Fixed token sequences used by the state machine.
        self.suffix_ids: List[int] = []
        self.para_sep_ids: List[int] = []
        self.after_val_sep_ids: List[int] = []
        self.after_val_end_ids: List[int] = []

        self.fixed_token_ids: List[int] = []
        self.fixed_token_index: int = 0

        # A string token can close a value and also emit punctuation after it.
        self.pending_after_val_text: str = ""
        self.string_value_complete = False
