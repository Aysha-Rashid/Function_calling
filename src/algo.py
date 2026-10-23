import argparse
from llm_sdk import Small_LLM_Model
from src.schema import FunctionDefinition, Prompts, Tokenizer
from typing import Dict, Tuple, List
import json


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


# def decoding_constraint(prompt: str, functions: FunctionDefinition, vocab_map: Dict[int, str], model: Small_LLM_Model) -> Dict[str, Any]:




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
    merge_path = model.get_path_to_merges_file()
    tokenizer_path = model.get_path_to_tokenizer_file()
    print(merge_path)
    print(tokenizer_path)
    lister = Tokenizer(vocab_map)
    print(lister.encoder("hello"))
    # prefix = ()

    #prefix is something that has the defined prompts that needs to be used?
    for each_prompt in prompts:
        # convert prompt into string
        # convert function definition into string
        string_each_prompt = each_prompt.prompt
        # string_list_functions = [each_function for each_function in functions]
        # print("checking each function:", type(string_list_functions))
        # print("checking each string:", string_each_prompt)
        # result = decoding_constraint(string_each_prompt, functions, vocab_map, model)
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
    model = Small_LLM_Model()
    vocab_map = get_the_vocab(model)
    #start for each prompt
    result = start_decoding_for_each_prompt(function, prompt, model, vocab_map)
    try:
        with open(arg.output, "w", encoding="utf-8") as output:
            output = result
    except Exception as e:
        print("Error in output file:", e)
    # print(vocab_map)
