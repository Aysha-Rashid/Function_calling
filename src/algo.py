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
def prefix_token_id(logits, partial_text, required_text, vocab_map):
    valid_token_ids = []

    for token_id, token_str in vocab_map.items():
        combined = partial_text + token_str

        if required_text.startswith(combined):
            valid_token_ids.append(token_id)

    # Apply the same logit-masking approach as prefix_token_id().
    masked_logits = np.full_like(logits, -np.inf)

    for token_id in valid_token_ids:
        masked_logits[token_id] = logits[token_id]

    return masked_logits


def function_token_id(partial: str, logits: list[float], current_text: str, vocab_map: dict[int, str], target: dict[str, str]):
    valid_ids = []
    for token_id, token_str in vocab_map.items():
        # print(f"key: {target.keys()}")
        for key in target.keys():
            # print(f"partial: {partial} + token_str: {token_str}")
            combined = partial + token_str
            if (key + '"').startswith(combined):
                print("partial + token_str: if the key is correct:")
                print(f"-----------------key:{key}----------------")
                print(f"-----------------Combined(partial + token_str): {partial} + {token_str}")
                valid_ids.append(token_id)
                break
    # saving the token_id from the logits list
    saving_logit = {token_id: logits[token_id] for token_id in valid_ids}
    # making everything in the logits list as -infinite
    logits[:] = [float('-inf')] * len(logits)
    if valid_ids:
        for token_id, scores in saving_logit.items():
            print(f"token_id:{token_id}, token_str:{vocab_map[token_id]}")
            logits[token_id] = scores
    else:
        print("no valid token found")
    return logits


# def masked_logit_logic(logits: list[float], valid_ids: list[int]) -> list[float]:
#     """

#     """
    


def adding_constraint(logits: list[float], state_json: DecodingStateJson, current_str: str, vocab_map: Dict[int, str], functions: FunctionDefinition) -> list[float]:
    """

    """
    print("checking the current state:", state_json.phase)
    name_prefix = '"name": "'
    parameter_prefix = ',\n"    parameters":{'
    valid_index = []
    if (state_json.phase == CurrentState.FUNCTION_NAME):
        partial_name = current_str.split(name_prefix)[-1] 
        # here the double qoute in the beginning is not being passed and its alright
        print(f"---------partial_name: {partial_name}----------")
        if not partial_name.endswith('"'):
            return function_token_id(partial_name, logits, current_str, vocab_map, state_json.valid_function_name)
        state_json.phase = CurrentState.AFTER_FUNCTION_NAME
        print("\n\n\nThe function name ended with double qoutes")
        print(f"function name: {partial_name}")
        state_json.selected_function = partial_name[:-1]
        # print(f"-------------- selected function name: {state_json.selected_function}------------")
        selected_function_object = state_json.valid_function_name.get(state_json.selected_function)
        # print(f"\n\n\nthe object of the selected_function: {selected_function_object}")
        state_json.parameter_key = dict(selected_function_object.parameters).keys()
        print(f"checking the parameters_key: {state_json.parameter_key}")
        state_json.parameter_value = dict(selected_function_object.parameters).values()
        print(f"checking the parameters_value: {state_json.parameter_value}")

        # state_json.parameter_values = 
        # i think i might have to put ',\n' in here to continue to the next step
        # return (masked_logit_logic(logits, valid_ids))
        # return masked_logit
    elif (state_json.phase == CurrentState.AFTER_FUNCTION_NAME):
        print("\n\n\n Comes here after choosing the function name")
        return prefix_token_id(logits, current_str, parameter_prefix, vocab_map)



def decoding_constraint(prompt: str, functions: FunctionDefinition, vocab_map: Dict[int, str], model: Small_LLM_Model, state_json: DecodingStateJson) -> Dict[str, Any]:
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
    # state_json = DecodingStateJson()
    generated : List[int] = []
    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        current_str = model.decode(generated)
        masked_logits = adding_constraint(logits, state_json, current_str, vocab_map, functions)
        # print("checking for token id:", logits)
        next_token_id = int(np.argmax(masked_logits))
        generated.append(next_token_id)
        print("decode:", model.decode(generated))
        print("prompt:", prompt)
        # break
        


def start_decoding_for_each_prompt(
    functions: FunctionDefinition,
    prompts: Prompts,
    model: Small_LLM_Model,
    vocab_map: Dict[int, str],
    state_json: DecodingStateJson
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
    # for each_prompt in prompts:
    #     # convert prompt into string
    #     # convert function definition into string
    #     string_each_prompt = each_prompt.prompt
    #     # string_list_functions = [each_function for each_function in functions]
    #     # print("checking each function:", type(string_list_functions))
    #     # print("checking each string:", string_each_prompt)
    #     result = decoding_constraint(string_each_prompt, functions, vocab_map, model)
    # # return result
    prompt_test = "What is the sum of 2 and 3?"
    result = decoding_constraint(prompt_test, functions, vocab_map, model, state_json)


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
    print("########checking initial function object########")
    valid_function_name = {func.name: func for func in function}
    state_json = DecodingStateJson(Functions_list=function)
    # state_json.Functions_list = function
    # print(f"checking the function_list in: {state_json.Functions_list}" )
    # print(f"function_name: {state_json.valid_function_name.keys()}")
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
    result = start_decoding_for_each_prompt(function, prompt, model, vocab_map, state_json)
    # try:
    #     with open(arg.output, "w", encoding="utf-8") as output:
    #         output = result
    # except Exception as e:
    #     print("Error in output file:", e)
    # # print(vocab_map)
