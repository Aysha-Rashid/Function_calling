import argparse
from llm_sdk import Small_LLM_Model
from src.schema import FunctionDefinition, Prompts, Tokenizer, DecodingStateJson, CurrentState, functions_to_call_schema
from typing import Dict, Tuple, List, Any
import json
import numpy as np
import math


def parse_arg() -> argparse.Namespace:
    """
        Parse command-line arguments
    Returns:
        The parsed input, Function-definition, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Call_me_maybe argument parsing"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    return parser.parse_args()


def get_the_vocab(model : Small_LLM_Model) -> Dict[int, str] :
    """
        Get the path of the vocabularies from the model and store them in Dict form
    Returns:
        The Dict of the tokens id and tokens str
    """
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r", encoding="utf-8") as vocabularies:
            raw_json: Dict[int, str] = json.load(vocabularies)
    except Exception as e:
        print("Error in the vocab map:", e)
    inverted = {}
    for token_str, token_id in raw_json.items():
        inverted[token_id] = token_str
    return inverted


# def masked_tokens(masked_logit: list[float], valid_token: list[int]) -> list[int]:
#     for each_token in masked_logit:
#         if masked_logit[each_token] == valid_token:
#             new_list[valid_token] = masked_logit[each_token]
#         else
#             new_list[each_token] == 
def prefix_token_id(prefix: str, logits: list[float], current_text: str, vocab_map: dict[int, str], model: Small_LLM_Model):
    for token_id, token_str in vocab_map.items():
        print("checking the token_str:", token_str)
        if token_str == prefix:
            print("come here bro")


def adding_constraint(logits: list[float], state_json: DecodingStateJson, current_str: str, vocab_map: Dict[int, str], model: Small_LLM_Model) -> list[float]:
    """

    """
    print("checking the current state:", state_json.phase)
    name_prefix = '"name": '
    parameter_prefix = ',\n"    parameters":{'
    
    if (state_json.phase == CurrentState.FUNCTION_NAME):
        # if current_str is None:
        partial_name = current_str.split(name_prefix)[-1]
        valid_index = []
        print(partial_name)
        if partial_name.endswith('"'):
            for token_id, token_str in vocab_map.items():
                if partial_name == token_str:
                #     valid_index.append(token_id)
                    print("what is the valid_index in here:", valid_index)
        prefix_token_id(name_prefix, logits, current_str, vocab_map, model)
        # return masked_logit
           


def decoding_constraint(prompt: str, functions: FunctionDefinition, vocab_map: Dict[int, str], model: Small_LLM_Model) -> Dict[str, Any]:
    """
        token-level state decoding contraint
    """
    func_descriptions = "\n".join(
        (
            f"- {function.name}: {function.description}\n"
            "  Parameters:\n"
            + "\n".join(
                f"    - {param_name}: {param_schema.type}"
                for param_name, param_schema in function.parameters.items()
            )
        )
        for function in functions
    )
    # print("check the list of functions: ", func_descriptions)
    user_prompt = (
        "You are a function calling model.\n"
        "Using the list of function below:"
        f"{func_descriptions} to answer the user query:"
        f"{prompt}\n"
        "and output the answer in this format:"
        "{\n"
        f' "prompt": "{prompt}",\n "name": "'
    )
    # print("does this work? ", user_prompt)
    # state_json = DecodingStateJson()
    input_ids = model.encode(user_prompt)[0].tolist()
    state_json = DecodingStateJson()
    generated : List[int] = []
    for _ in range(100):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        current_str = model.decode(generated)
        masked_logit = adding_constraint(logits, state_json, current_str, vocab_map, model)
        # print("checking for token id:", logits)
        # next_token_id = int(np.argmax(masked_logits))
        # generated.append(next_token_id)
        # print("decode:", model.decode(generated))
        break
        


def start_decoding_for_each_prompt(
    functions: FunctionDefinition,
    prompts: Prompts,
    model: Small_LLM_Model,
    vocab_map: Dict[int, str]
    ) -> None:
    """Start decoding for each prompt"""
    # prefix?? caches?? encoing?? decoding??
    # what are the things i should consider caching? 
        # the function_definition (schema information)
        # the decoding schema for each prompt
        # vocab_map
        # reuse prompt encode

    # functions_definition
    # merge_path = model.get_path_to_merges_file()
    # tokenizer_path = model.get_path_to_tokenizer_file()
    # print(merge_path)
    # print(tokenizer_path)
    # lister = Tokenizer(vocab_map)
    # print(lister.encoder("hello"))
    # prefix = ()

    #prefix is something that has the defined prompts that needs to be used?
    for each_prompt in prompts:
        # convert prompt into string
        # convert function definition into string
        string_each_prompt = each_prompt.prompt
        # string_list_functions = [each_function for each_function in functions]
        # print("checking each function:", type(string_list_functions))
        # print("checking each string:", string_each_prompt)
        result = decoding_constraint(string_each_prompt, functions, vocab_map, model)
    # return result


def store_the_functions(arg: argparse.Namespace) -> tuple[list[FunctionDefinition], list[Prompts]]:
    """
        Store the content of the files in an object of classes
    Returns:
        functionDefinition object and prompt object
    """
    try: 
        with open(arg.functions_definition, "r", encoding="utf-8") as func_file:
            raw_func: Dict[str, str] = json.load(func_file)
            function_def = [
                FunctionDefinition(**each_func)
                for each_func in raw_func
            ]
        with open(arg.input, "r", encoding="utf-8") as prompt_file:
            raw_prompt = json.load(prompt_file)
            prompt = [
                Prompts(**each_prompt)
                for each_prompt in raw_prompt
            ]
    except Exception as e:
        print("Error in function file:", e)
    return function_def, prompt


#here we are going to read the files and store their content in an object of their class
def initialize_everything() -> None:
    """
        Intialize the object of all the classes with the content of the files
    """
    arg = parse_arg()
    function, prompt = store_the_functions(arg)
    # call_schema = functions_to_call_schema(function)
    # print(json.dumps(call_schema, indent=2))
    # valid_call = {
    #     "name": "fn_add_numbers",
    #     "parameters": {"a": 2, "f": 3},
    # }
    # invalid_call = {
    #     "name": "fn_add_numbers",
    #     "parameters": {"a": 2, "c": 3},
    # }
    # validate_function_call(valid_call, function)
    # print("Valid call accepted")
    model = Small_LLM_Model()
    vocab_map = get_the_vocab(model)
    #start for each prompt
    result = start_decoding_for_each_prompt(function, prompt, model, vocab_map)
    # try:
    #     with open(arg.output, "w", encoding="utf-8") as output:
    #         output = result
    # except Exception as e:
    #     print("Error in output file:", e)
    # # print(vocab_map)
