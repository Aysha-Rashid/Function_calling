import argparse
from llm_sdk import Small_LLM_Model
from typing import Dict
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
    print(vocab_path)
    try:
        with open(vocab_path, "r", encoding="utf-8") as vocabularies:
            raw_json: Dict[int, str] = json.load(vocabularies)
            print(raw_json)
    except Exception as e:
        print("Error in the vocab map:", e)
    # return raw_json


#here we are going to read the files and store their content in an object of their class
def initialize_everything() -> None:
    """
        Intialize the object of all the classes with the content of the files
    """
    arg = parse_arg()
    model = Small_LLM_Model()
    vocab_map = get_the_vocab(model)
    # print(vocab_map)