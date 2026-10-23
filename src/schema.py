from typing import Any, Dict
from pydantic import base
from dataclass import dataclass, field


class FunctionDefinition(Base):
    name: str,
    description: str,
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class Prompts(Base):
    prompt: str


class DecodingStateJson(Base):
    Function_name: str,
    After_Function_name: str,
    Parameter_name: str,
    After_Parameter_name: str,
    Parameter_key: str,
    Parameter_value: str,
    Parameter_partial: str,
    End: str


@dataclass
class stateMachine():
    