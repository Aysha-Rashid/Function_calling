import argparse
from llm_sdk import Small_LLM_Model
from src.schema import FunctionDefinition, Prompts, Tokenizer, DecodingStateJson, CurrentState, functions_to_call_schema
from typing import Dict, Tuple, List, Any
import json
import numpy as np
import math
import re

# JSON number grammar: prefix and completion are deliberately different.
NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
NUMBER_PREFIX_RE = re.compile(
    r"(?:-|(?:-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?))?\Z"
)


def constrain_number_value(logits, vocab_map, state_json):
    """Mask tokens to prefixes of JSON numbers or a legal final delimiter."""
    current = state_json.parameter_value_generated
    delimiter = ("}" if state_json.parameter_index ==
                 len(state_json.parameter_key) - 1 else ",")
    valid = []
    complete = NUMBER_RE.fullmatch(current) is not None
    for token_id, token in vocab_map.items():
        if not token or token_id >= len(logits):
            continue
        candidate = current + token
        if NUMBER_PREFIX_RE.fullmatch(candidate) and candidate != current:
            valid.append(token_id)
        elif complete and token == delimiter:
            valid.append(token_id)
        elif candidate.endswith(delimiter) and NUMBER_RE.fullmatch(candidate[:-1]):
            valid.append(token_id)
    if not valid:
        raise ValueError("No valid JSON number continuation")
    return contraint_mask(logits, valid)


# Keep the original name available to callers during migration.
constrain_integer_value = constrain_number_value


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


def contraint_mask(logits: list[float], valid_ids: list[int]) -> list[float]:
    """
        Mask the logits based on the valid token ids
    Args:
        logits: The list of logits from the model
        valid_ids: The list of valid token ids
    Returns:
        The masked logits
    """
    # making everything in the logits list as -infinite
    masked_logit = []
    masked_logit[:] = [float('-inf')] * len(logits)
    if valid_ids:
        for token_id in valid_ids:
            masked_logit[token_id] = logits[token_id]  # or any other value you want to assign to valid tokens
    else:
        print("no valid token found")
    return masked_logit


def contraint_prefix_mask(logits: list[float], valid_ids: list[int], state_json: DecodingStateJson) -> list[float]:
    masked_logit = []
    masked_logit[:] = [float('-inf')] * len(logits)
    if state_json.prefix_index >= len(valid_ids):
        return logits
    token_id = valid_ids[state_json.prefix_index]
    masked_logit[token_id] = logits[token_id]
    state_json.prefix_index += 1
    return masked_logit


def function_token_id(partial: str, logits: list[float], current_text: str, vocab_map: dict[int, str], target: dict[str, str]):
    valid_ids = []
    for token_id, token_str in vocab_map.items():
        for key in target.keys():
            combined = partial + token_str
            if (key + '"').startswith(combined):
                valid_ids.append(token_id)
                break
    return (contraint_mask(logits, valid_ids))


def constrain_boolean_value(logits, vocab_map, state_json):
    current_value = state_json.parameter_value_generated
    valid_token_ids = []
    is_last_parameter = (
        state_json.parameter_index == len(state_json.parameter_key) - 1
    )
    for token_id, token in vocab_map.items():
        candidate = current_value + token
        if "true".startswith(candidate) or "false".startswith(candidate):
            valid_token_ids.append(token_id)
        elif current_value in ["true", "false"]:
            if not is_last_parameter and token == ",":
                valid_token_ids.append(token_id)
            elif is_last_parameter and token == "}":
                valid_token_ids.append(token_id)
    return contraint_mask(logits, valid_token_ids)


def adding_constraint(logits: list[float], state_json: DecodingStateJson, current_str: str, vocab_map: Dict[int, str], model: Small_LLM_Model, functions: FunctionDefinition) -> list[float]:
    """

    """
    print("checking the current state:", state_json.phase)
    name_prefix = '"name": "'
    parameter_prefix = ',\n "parameters": {"'
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
        state_json.prefix_id = model.encode(parameter_prefix)
        state_json.prefix_index = 0
        state_json.prefix_size = len(state_json.prefix_id[0])
        selected_function_object = state_json.valid_function_name.get(state_json.selected_function)
        state_json.parameter_key = list(selected_function_object.parameters.keys())
        print(f"checking the parameters_key: {state_json.parameter_key}")
        state_json.parameter_value = list(selected_function_object.parameters.values())
        print(f"checking the parameters_value: {state_json.parameter_value}")
    if (state_json.phase == CurrentState.AFTER_FUNCTION_NAME):
        if state_json.prefix_size > 0:
            # prefix_id = model.encode(parameter_prefix)
            state_json.prefix_size -= 1
            return contraint_prefix_mask(logits, state_json.prefix_id[0].tolist(), state_json)
        else:
            # prefix_size in here depends on the size of the key of first in the list
            print("the datatype of parameter key:", type(state_json.parameter_key[0]))
            state_json.prefix_size = 4
            state_json.prefix_index = 0
            state_json.phase = CurrentState.PARAMETER_KEY
    if (state_json.phase == CurrentState.PARAMETER_KEY):
        print("-----------------------Parameter_key--------------------")
        current_key = state_json.parameter_key[state_json.parameter_index]
        print("the current parameter_key:", current_key)
        if not state_json.parameter_key_ids:
            if state_json.parameter_index == 0:
                key_text = current_key + '": '
            else:
                key_text = '"' + current_key + '": '
            state_json.parameter_key_ids = model.encode(key_text)[0].tolist()
            state_json.prefix_index = 0
        if state_json.prefix_index < len(state_json.parameter_key_ids):
            return contraint_prefix_mask(
                logits, state_json.parameter_key_ids, state_json
            )
        state_json.parameter_key_ids = []
        state_json.prefix_index = 0
        state_json.phase = CurrentState.PARAMETER_VALUE
    if state_json.phase == CurrentState.PARAMETER_VALUE:
        print("---------------------------Parameter_value-------------------------")
        current_type = state_json.parameter_value[state_json.parameter_index]
        print("checking the type of the parameter: ", current_type.type)
        print("the current parameter_key:", state_json.parameter_key[state_json.parameter_index])
        # A completed value is followed by the next key or by END.
        if getattr(state_json, "value_finished", False):
            state_json.value_finished = False
            state_json.parameter_value_generated = ""
            state_json.string_started = False
            state_json.string_closed = False
            state_json.string_quote_ids = []
            state_json.string_content_token_ids = []
            state_json.parameter_key_ids = []
            state_json.prefix_index = 0
            if state_json.parameter_index == len(state_json.parameter_key) - 1:
                state_json.phase = CurrentState.END
                return logits
            state_json.parameter_index += 1
            state_json.phase = CurrentState.PARAMETER_KEY
            return adding_constraint(
                logits, state_json, current_str, vocab_map, model, functions
            )
        if current_type.type == "number":
            return constrain_number_value(logits, vocab_map, state_json)
        elif current_type.type == "string":
            print("for string it comes here")
            return constrain_string_value(logits, vocab_map, state_json, model)
        elif current_type.type == bool:
            return constrain_boolean_value(logits, vocab_map, state_json)


def is_valid_json_string_content(value: str) -> tuple[bool, bool]:
    """
    Returns:
        (valid_prefix, can_close)
    """
    i = 0

    while i < len(value):
        char = value[i]
        if ord(char) < 0x20:
            return False, False
        if char == '"':
            return False, False
        if char == '\\':
            i += 1
            # An incomplete escape is a valid prefix,
            # but the string cannot close yet.
            if i >= len(value):
                return True, False
            if value[i] == 'u':
                remaining = value[i + 1:i + 5]
                if any(c not in "0123456789abcdefABCDEF" for c in remaining):
                    return False, False
                if len(remaining) < 4:
                    return True, False
                i += 4
            elif value[i] not in '"\\/bfnrt':
                return False, False
        i += 1
    return True, True


def _string_fragment_status(value: str) -> tuple[bool, bool, str, str]:
    """Return (valid, closed, content, suffix) for a JSON string fragment."""
    if not value.startswith('"'):
        return False, False, "", ""
    escaped = False
    for index in range(1, len(value)):
        char = value[index]
        if char == '"' and not escaped:
            content = value[1:index]
            valid, can_close = is_valid_json_string_content(content)
            suffix = value[index + 1:]
            return valid and can_close and suffix in ("", ",", "}"), True, content, suffix
        if char == "\\":
            escaped = not escaped
        else:
            escaped = False
    content = value[1:]
    valid, _ = is_valid_json_string_content(content)
    return valid, False, content, ""


def constrain_string_value(logits, vocab_map, state_json, model):
    """Permit arbitrary JSON string content, followed by the required delimiter.

    Decode only tokens with finite logits. Exact sequence decoding is retained
    because vocabulary spellings are not guaranteed to be decoded text.
    """
    if not state_json.string_started:
        if not state_json.string_quote_ids:
            state_json.string_quote_ids = model.encode('"')[0].tolist()
            state_json.prefix_index = 0
        if state_json.prefix_index < len(state_json.string_quote_ids):
            return contraint_prefix_mask(
                logits, state_json.string_quote_ids, state_json
            )
        state_json.string_content_token_ids = list(state_json.string_quote_ids)
        state_json.string_started = True
        state_json.string_quote_ids = []
        state_json.prefix_index = 0

    delimiter = ("}" if state_json.parameter_index ==
                 len(state_json.parameter_key) - 1 else ",")
    current_ids = state_json.string_content_token_ids
    current = model.decode(current_ids)
    valid_current, closed_current, _, suffix_current = _string_fragment_status(current)
    if not valid_current or suffix_current:
        raise ValueError("Invalid string generation state")

    valid_ids = []
    for token_id in vocab_map:
        if token_id >= len(logits) or not math.isfinite(float(logits[token_id])):
            continue
        decoded = model.decode(current_ids + [token_id])
        valid, closed, _, suffix = _string_fragment_status(decoded)
        if not valid or decoded == current:
            continue
        if closed_current:
            if closed and suffix == delimiter:
                valid_ids.append(token_id)
        elif not closed or suffix in ("", delimiter):
            valid_ids.append(token_id)
    if not valid_ids:
        raise ValueError("No valid JSON string continuation")
    return contraint_mask(logits, valid_ids)


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
    json_begin_prefix = ("{\n"
        f' "prompt": "{prompt}",\n "name": "')
    user_prompt = (
        "You are a function calling model.\n"
        "Using the list of function below:"
        f"{func_descriptions} to answer the user query:"
        f"{prompt}\n"
        "and output the answer in this format:"
        f"{json_begin_prefix}"
    )
    # print("does this work? ", user_prompt)
    # state_json = DecodingStateJson()
    input_ids = model.encode(user_prompt)[0].tolist()
    # state_json = DecodingStateJson()
    generated : List[int] = []
    for _ in range(100):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        current_str = model.decode(generated)
        print("current_str:", repr(current_str))
        masked_logits = adding_constraint(logits, state_json, current_str, vocab_map, model, functions)
        if (state_json.phase == CurrentState.END):
            break
        # print("checking for token id:", logits)
        next_token_id = int(np.argmax(masked_logits))
        generated.append(next_token_id)
        decoded_after_token = model.decode(generated)
        # Decode in sequence context. A single token's decoded spelling can
        # differ from its contribution after the preceding tokens.
        token_text = decoded_after_token[len(current_str):]
        if state_json.phase == CurrentState.PARAMETER_VALUE:
            current_type = state_json.parameter_value[state_json.parameter_index]
            delimiter = ("}" if state_json.parameter_index ==
                         len(state_json.parameter_key) - 1 else ",")
            if current_type.type == "string" and state_json.string_started:
                state_json.string_content_token_ids.append(next_token_id)
                decoded_content = model.decode(state_json.string_content_token_ids)
                valid, closed, content, suffix = _string_fragment_status(decoded_content)
                if not valid:
                    raise ValueError("Invalid generated string")
                state_json.parameter_value_generated = content
                state_json.string_closed = closed
                if closed and suffix == delimiter:
                    state_json.value_finished = True
            elif current_type.type == "number":
                state_json.parameter_value_generated += token_text
                if state_json.parameter_value_generated.endswith(delimiter):
                    state_json.parameter_value_generated = (
                        state_json.parameter_value_generated[:-1]
                    )
                    state_json.value_finished = True
        print("decode:", model.decode(generated))
        print("prompt:", prompt)
        # if model.decode(generated).endswith("}") :
        #     print("The json format ended")
        #     break
        # break
    combined = json_begin_prefix + model.decode(generated)
    print(combined)
    # combined the prompt, name prefix with the generated list  


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
    prompt_test = "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
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
