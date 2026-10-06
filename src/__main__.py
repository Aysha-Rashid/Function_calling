import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from llm_sdk import Small_LLM_Model
from .constraint_functions import adding_constraint, after_val_suffix
from .schema import FunctionDefinition, GenerationState, JSONStructure, Prompt
from .utils import closes_string, is_complete_value


DEBUG = os.environ.get("CMM_DEBUG") == "1"
class FixedTokens:
    """Tokenized JSON fragments reused by every prompt."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self.suffix_ids = model.encode(
            ',\n  "parameters": {\n    "'
        )[0].tolist()
        self.para_sep_ids = model.encode(": ")[0].tolist()
        self.after_val_sep_ids = model.encode(',\n    "')[0].tolist()
        self.after_val_end_ids = model.encode("\n  }\n}")[0].tolist()


def parse_arg() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Call_me_maybe argument parser"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        # type=str,
        help="Path to the function_calling_tests input file",
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        # type=str,
        help="Path to the functions_definition input file",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        # type=str,
        help="Path to the output file",
    )
    return parser.parse_args()


def load_func_and_prompt(
    functions_definition: str,
    prompts: str,
) -> tuple[list[FunctionDefinition], list[Prompt]]:
    """Load function definitions and prompts from JSON files."""
    try:
        with open(functions_definition, "r", encoding="utf-8") as function_file:
            raw_function_definitions = json.load(function_file)
            object_function_definitions = [
                FunctionDefinition(**item)
                for item in raw_function_definitions
            ]

        with open(prompts, "r", encoding="utf-8") as prompts_file:
            raw_prompts = json.load(prompts_file)
            object_prompts = [Prompt(**item) for item in raw_prompts]
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Error loading files:\n"
            f"functions_definition={functions_definition!r}\n"
            f"prompts={prompts!r}\n"
            f"original_error={exc}"
        ) from exc

    return object_function_definitions, object_prompts


def load_inverted_tokens(model: Small_LLM_Model) -> Dict[int, str]:
    """Return a token_id -> decoded token-text mapping."""
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r", encoding="utf-8") as file:
            raw_json_vocab: Dict[str, int] = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Error loading vocabulary from {vocab_path!r}: {exc}"
        ) from exc

    token_ids = sorted(raw_json_vocab.values())
    return {
        token_id: model.decode([token_id])
        for token_id in token_ids
    }


def _configure_state(
    schema: JSONStructure,
    fixed_tokens: FixedTokens,
) -> None:
    """Attach reusable fixed token sequences to a prompt's state."""
    schema.suffix_ids = fixed_tokens.suffix_ids
    schema.para_sep_ids = fixed_tokens.para_sep_ids
    schema.after_val_sep_ids = fixed_tokens.after_val_sep_ids
    schema.after_val_end_ids = fixed_tokens.after_val_end_ids


def decode_constraint(
    prompt_obj: Prompt,
    vocab_map: Dict[int, str],
    functions: List[FunctionDefinition],
    model: Small_LLM_Model,
    fixed_tokens: FixedTokens,
    prefix_cache: Dict[tuple[frozenset[str], str], List[int]], # what does prefix cache do
) -> Dict[str, Any]:
    """Generate one constrained function call."""
    state_json = JSONStructure(functions, prefix_cache)
    _configure_state(state_json, fixed_tokens)

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
    prefix_str = (
        "{\n"
        f'  "prompt": {json.dumps(prompt_obj.prompt)},\n'
        '  "name": "'
    )
    system_prompt = (
        "You are a function calling model.\n"
        "Choose the correct function and extract its arguments from the user's prompt.\n"
        "Do not answer or calculate the user's question.\n"
        "For numeric arguments, copy the numbers stated in the "
        "user's prompt exactly as they appear."
        "Available Functions:\n"
        f"{func_descriptions}\n"
        f"User Prompt: {prompt_obj.prompt}\n"
        "JSON Output:\n"
        f"{prefix_str}"
    )
    input_ids = model.encode(system_prompt)[0].tolist()
    generated_ids: List[int] = []
    generated_text = ""
    vocab_items: Sequence[Tuple[int, str]] = tuple(vocab_map.items())
    for _ in range(120):
        if state_json.state == GenerationState.END:
            break
        cur_str = prefix_str + generated_text
        logits = model.get_logits_from_input_ids(input_ids + generated_ids)
        masked_logits = adding_constraint(cur_str, logits, vocab_items, state_json)
        if state_json.state == GenerationState.END: # why again? ig incase once it comes to adding_constraint and it changes state to end
            break
        if not np.isfinite(masked_logits).any():
            raise ValueError(f"No valid token available in state {state_json.state}")
        next_token_id = int(np.argmax(masked_logits))
        generated_ids.append(next_token_id)
        token_text = model.decode([next_token_id])
        generated_text += token_text
        if state_json.state == GenerationState.AFTER_FUNCTION_NAME:
            state_json.fixed_token_index += 1
        elif state_json.state == GenerationState.AFTER_PARAMETER_KEY:
            state_json.fixed_token_index += 1
        elif state_json.state == GenerationState.AFTER_PARAMETER_VAL:
            if state_json.pending_after_val_text:
                if not state_json.pending_after_val_text.startswith(token_text):
                    raise ValueError(
                        "Generated token does not match pending fixed text"
                    )
                state_json.pending_after_val_text = (
                    state_json.pending_after_val_text[len(token_text):])
            else:
                state_json.fixed_token_index += 1
        elif state_json.state == GenerationState.PARAMETER_KEY:
            state_json.partial_para += token_text
        elif state_json.state == GenerationState.PARAMETER_VAL:
            remainder = None
            if state_json.cur_para_type == "string":
                remainder = closes_string(state_json.part_para_val, token_text)
            if remainder is not None:
                consumed_len = len(token_text) - len(remainder)
                state_json.part_para_val += token_text[:consumed_len]
                state_json.pending_after_val_text = (
                    after_val_suffix(state_json)[len(remainder):])
                state_json.state = GenerationState.AFTER_PARAMETER_VAL
            else:
                value_is_complete = is_complete_value(
                    state_json.cur_para_type,
                    state_json.part_para_val)
                if (value_is_complete
                    and next_token_id == state_json.after_val_sep_ids[0]):
                    state_json.fixed_token_ids = state_json.after_val_sep_ids
                    state_json.fixed_token_index = 1
                    state_json.state = GenerationState.AFTER_PARAMETER_VAL
                elif (value_is_complete
                    and next_token_id == state_json.after_val_end_ids[0]):
                    state_json.fixed_token_ids = state_json.after_val_end_ids
                    state_json.fixed_token_index = 1
                    state_json.state = GenerationState.AFTER_PARAMETER_VAL

                else:
                    state_json.part_para_val += token_text

        if DEBUG:
            print(
                f"state={state_json.state.value} "
                f"token={next_token_id} "
                f"text={token_text!r} "
                f"value={state_json.part_para_val!r}"
            )

    raw_json = prefix_str + generated_text

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Generated output is not valid JSON"
        ) from exc


def main() -> None:
    args = parse_arg()
    model = Small_LLM_Model()
    functions, prompts = load_func_and_prompt(args.functions_definition, args.input)
    vocab_map = load_inverted_tokens(model)
    prefix_cache: Dict[tuple[frozenset[str], str], List[int]] = {}
    fixed_tokens = FixedTokens(model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    for each_prompt in prompts:
        try:
            result_dict = decode_constraint(
                each_prompt,
                vocab_map,
                functions,
                model,
                fixed_tokens,
                prefix_cache,
            )
        except Exception as exc:
            failures.append(each_prompt.prompt)
            if DEBUG:
                print(f"Failed for {each_prompt.prompt!r}: {exc}")
            continue
        results.append(result_dict)
    with open(args.output, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)
    print(f"Successfully saved output to {args.output}")
    print(f"Succeeded: {len(results)}/{len(prompts)}")
    if failures:
        print(f"Failed prompts: {failures}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
