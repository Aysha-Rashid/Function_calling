from enum import Enum
from typing import Any, Dict, List, Set
from pydantic import BaseModel
from dataclasses import dataclass, field


class ParameterSchema(BaseModel):
    """Describe the type of a function parameter."""
    type: str


class FunctionDefinition(BaseModel):
    """Describe a callable function exposed to the language model."""
    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: Dict[str, Any]


class Prompt(BaseModel):
    """Represent one user prompt that requires a function call."""
    prompt: str


class Result(BaseModel):
    """Represent a generated function-call result."""
    prompt: str
    name: str
    parameters: Dict[str, Any]


class FunctionCallResult(BaseModel):
    """Represent the prompt, function name, and arguments of a call."""
    prompt: str
    name: str
    parameters: Dict[str, Any]


class GenerationState(Enum):
    """Represent the stages of token-by-token constrained generation."""
    FUNCTION_NAME = "FUNCTION_NAME"
    AFTER_FUNCTION_NAME = "AFTER_FUNCTION_NAME"
    PARAMETER_KEY = "PARAMETER_KEY"
    AFTER_PARAMETER_KEY = "AFTER_PARAMETER_KEY"
    PARAMETER_VAL = "PARAMETER_VAL"
    AFTER_PARAMETER_VAL = "AFTER_PARAMETER_VAL"
    END = "END"


@dataclass
class JSONStructure:
    """Store mutable state for token-by-token constrained generation.

    Args:
        functions: Function definitions available during generation.
        prefix_cache: Shared cache of valid name-prefix token IDs.
    """
    functions: List[FunctionDefinition]
    prefix_cache: Dict[
        tuple[frozenset[str], str],
        List[int],
    ] = field(default_factory=dict)
    functions_by_name: Dict[str, FunctionDefinition] = field(init=False)
    valid_func_names: Set[str] = field(init=False)
    param_keys_by_func: Dict[str, Set[str]] = field(init=False)
    state: GenerationState = GenerationState.FUNCTION_NAME
    selected_function: str | None = None
    selected_function_object: FunctionDefinition | None = None
    remaining_parameters: Set[str] = field(default_factory=set)
    cur_para: str | None = None
    cur_para_type: str | None = None
    generated_parameters: Set[str] = field(default_factory=set)
    partial_para: str = ""
    part_para_val: str = ""
    suffix_ids: List[int] = field(default_factory=list)
    para_sep_ids: List[int] = field(default_factory=list)
    after_val_sep_ids: List[int] = field(default_factory=list)
    after_val_end_ids: List[int] = field(default_factory=list)
    fixed_token_ids: List[int] = field(default_factory=list)
    fixed_token_index: int = 0
    pen_after_val_text: str = ""
    string_value_complete: bool = False

    def __post_init__(self) -> None:
        """Build lookup tables derived from the supplied functions."""
        self.functions_by_name = {
            function.name: function
            for function in self.functions
        }
        self.valid_func_names = set(self.functions_by_name)
        self.param_keys_by_func = {
            function.name: set(function.parameters)
            for function in self.functions
        }
