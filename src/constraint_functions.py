import os
from typing import Dict, List, Sequence, Set
import numpy as np
from .schema import GenerationState, JSONStructure
from .utils import (
    STRING_COMPLETE,
    STRING_INCOMPLETE,
    apply_token_mask,
    closes_string,
    is_complete_number,
    is_complete_string,
    is_number_prefix,
    scan_string,
)


DEBUG = os.environ.get("CMM_DEBUG") == "1"
DEBUG_STRING_CONSTRAINT = os.environ.get("CMM_DEBUG_STRING") == "1"


def after_val_suffix(schema: JSONStructure) -> str:
    """Return the fixed JSON text required after the current value."""
    if schema.selected_function is None:
        raise ValueError("No function selected")

    if schema.remaining_parameters:
        return ',\n    "'
    return "\n  }\n}"


def constrain_fixed_tokens(
    logits: np.ndarray,
    token_ids: Sequence[int],
    index: int,
) -> np.ndarray:
    """Allow only the next token in a fixed token sequence."""
    if index >= len(token_ids):
        return logits

    masked_logits = np.full_like(logits, -float("inf"))
    token_id = token_ids[index]
    masked_logits[token_id] = logits[token_id]
    return masked_logits


def constrain_fixed_text(
    logits: np.ndarray,
    vocab_items: Sequence[tuple[int, str]],
    text: str,
) -> np.ndarray:
    """Allow tokens that can emit the next part of fixed text."""
    valid_indices = [
        token_id
        for token_id, token_text in vocab_items
        if token_text and text.startswith(token_text)
    ]

    if not valid_indices:
        raise ValueError(
            f"No token can produce the required text {text!r}"
        )

    return apply_token_mask(logits, valid_indices)


def _prefix_token_ids(
    partial: str,
    targets: Set[str],
    vocab_items: Sequence[tuple[int, str]],
    cache: Dict[tuple[frozenset[str], str], List[int]],
) -> List[int]:
    """Return token IDs that keep at least one target reachable.
    Results are cached because function and parameter names are reused across
    prompts and the same prefixes recur during autoregressive decoding.
    """
    key = (frozenset(targets), partial)
    cached = cache.get(key)
    if cached is not None:
        return cached

    valid_ids = [
        token_id
        for token_id, token_text in vocab_items
        if any(
            (target + '"').startswith(partial + token_text)
            for target in targets
        )
    ]
    cache[key] = valid_ids
    return valid_ids


def constrain_para_key(
    logits: np.ndarray,
    vocab_items: Sequence[tuple[int, str]],
    schema: JSONStructure,
) -> np.ndarray:
    """Constrain generation to valid parameter-name continuations."""
    if schema.selected_function is None:
        raise ValueError("No function selected")

    valid_indices = _prefix_token_ids(
        schema.partial_para,
        schema.remaining_parameters,
        vocab_items,
        schema.prefix_cache,
    )

    if not valid_indices:
        raise ValueError(
            "No valid parameter continuation.\n"
            f"Selected function: {schema.selected_function}\n"
            f"Allowed parameters: {schema.remaining_parameters}\n"
            f"Partial parameter: {schema.partial_para!r}"
        )

    return apply_token_mask(logits, valid_indices)


def _is_valid_string_step(
    partial_value: str,
    token_text: str,
    suffix: str,
) -> bool:
    """Return True when token_text legally continues the JSON string."""
    if not token_text:
        return False

    status_code, _ = scan_string(partial_value + token_text)

    if status_code == STRING_INCOMPLETE:
        return True
    if status_code != STRING_COMPLETE:
        return False

    remainder = closes_string(partial_value, token_text)
    return remainder is not None and suffix.startswith(remainder)


def _debug_string_constraint(
    schema: JSONStructure,
    partial_value: str,
    logits: np.ndarray,
    vocab_items: Sequence[tuple[int, str]],
    valid_indices: Sequence[int],
) -> None:
    """Print focused diagnostics for a stuck string value."""
    candidate = partial_value + '"'
    print(f"[string] current={partial_value!r}")
    print(f"[string] +quote={candidate!r}")
    print(f"[string] complete={is_complete_string(candidate)}")

    valid_set = set(valid_indices)
    for token_id, token_text in vocab_items:
        if token_text != '"':
            continue
        print(
            f"[string] quote id={token_id} "
            f"allowed={token_id in valid_set} "
            f"logit={logits[token_id]:.3f}"
        )

    top_legal = sorted(
        valid_indices,
        key=lambda token_id: logits[token_id],
        reverse=True,
    )[:8]
    print("[string] top legal continuations:")
    for token_id in top_legal:
        print(
            f"  id={token_id} {dict(vocab_items)[token_id]!r} "
            f"logit={logits[token_id]:.3f}"
        )


def constrain_para_val(
    logits: np.ndarray,
    vocab_items: Sequence[tuple[int, str]],
    schema: JSONStructure,
) -> np.ndarray:
    """Constrain a parameter value according to its declared type."""
    para_type = schema.cur_para_type
    partial_value = schema.part_para_val
    valid_indices: List[int] = []
    val_is_complete = False

    if para_type == "number":
        for token_id, token_text in vocab_items:
            candidate = partial_value + token_text
            if is_number_prefix(candidate):
                valid_indices.append(token_id)
        val_is_complete = is_complete_number(partial_value)

    elif para_type == "boolean":
        for token_id, token_text in vocab_items:
            candidate = partial_value + token_text
            if "true".startswith(candidate) or "false".startswith(candidate):
                valid_indices.append(token_id)
        val_is_complete = partial_value in {"true", "false"}

    elif para_type == "string":
        if not partial_value:
            opening_ids = [
                token_id
                for token_id, token_text in vocab_items
                if token_text == '"'
            ]
            if not opening_ids:
                raise ValueError("No opening quote token in vocabulary")
            return apply_token_mask(logits, opening_ids)

        suffix = after_val_suffix(schema)
        valid_indices = [
            token_id
            for token_id, token_text in vocab_items
            if _is_valid_string_step(partial_value, token_text, suffix)
        ]
        val_is_complete = is_complete_string(partial_value)

        if DEBUG_STRING_CONSTRAINT:
            _debug_string_constraint(
                schema,
                partial_value,
                logits,
                vocab_items,
                valid_indices,
            )

    else:
        raise ValueError(f"Unsupported parameter type: {para_type!r}")

    if val_is_complete:
        valid_indices.extend(_val_terminator_ids(schema))

    if not valid_indices:
        raise ValueError(
            "No valid value continuation.\n"
            f"Type: {para_type!r}\n"
            f"Value: {partial_value!r}"
        )

    return apply_token_mask(logits, list(dict.fromkeys(valid_indices)))


def _val_terminator_ids(schema: JSONStructure) -> List[int]:
    """Return the first token allowed after a complete value."""
    if schema.remaining_parameters:
        return schema.after_val_sep_ids[:1]
    return schema.after_val_end_ids[:1]


def adding_constraint(
    cur_str: str,
    logits: List[float],
    vocab_items: Sequence[tuple[int, str]],
    schema: JSONStructure,
) -> np.ndarray:
    """Apply token-level constraints according to the current state."""
    masked_logits = np.asarray(logits, dtype=np.float32) # asarray is to convert list to np.ndarray
    if schema.state == GenerationState.FUNCTION_NAME:
        name_prefix = '"name": "'
        part_name = cur_str.split(name_prefix)[-1]
        if part_name.endswith('"'):
            selected_name = part_name[:-1]
            if selected_name in schema.valid_func_names:
                schema.selected_function = selected_name
                schema.selected_function_object = next(
                    function
                    for function in schema.functions
                    if function.name == selected_name
                )
                schema.remaining_parameters = set(
                    schema.selected_function_object.parameters.keys())
                schema.state = GenerationState.AFTER_FUNCTION_NAME
                schema.fixed_token_ids = schema.suffix_ids
                schema.fixed_token_index = 0
                return constrain_fixed_tokens(masked_logits,
                                              schema.fixed_token_ids,
                                              schema.fixed_token_index)
        valid_indices = _prefix_token_ids(part_name, schema.valid_func_names,
                                          vocab_items, schema.prefix_cache)
        return apply_token_mask(masked_logits, valid_indices)

    if schema.state == GenerationState.AFTER_FUNCTION_NAME:
        if schema.fixed_token_index >= len(schema.fixed_token_ids):
            schema.state = GenerationState.PARAMETER_KEY
            schema.partial_para = ""
            return constrain_para_key(masked_logits, vocab_items, schema)
        return constrain_fixed_tokens(
            masked_logits,
            schema.fixed_token_ids,
            schema.fixed_token_index,
        )

    if schema.state == GenerationState.PARAMETER_KEY:
        if schema.partial_para.endswith('"'):
            parameter = schema.partial_para[:-1]
            if parameter in schema.remaining_parameters:
                schema.cur_para = parameter
                schema.generated_parameters.add(parameter)
                schema.remaining_parameters.remove(parameter)
                schema.part_para_val = ""

                if schema.selected_function_object is None:
                    raise ValueError("Selected function object is missing")

                schema.cur_para_type = (
                    schema.selected_function_object.parameters[parameter].type
                )
                schema.fixed_token_ids = schema.para_sep_ids
                schema.fixed_token_index = 0
                schema.state = GenerationState.AFTER_PARAMETER_KEY
                return constrain_fixed_tokens(
                    masked_logits,
                    schema.fixed_token_ids,
                    schema.fixed_token_index,
                )

        return constrain_para_key(masked_logits, vocab_items, schema)

    if schema.state == GenerationState.AFTER_PARAMETER_KEY:
        if schema.fixed_token_index >= len(schema.fixed_token_ids):
            schema.state = GenerationState.PARAMETER_VAL
            return constrain_para_val(masked_logits, vocab_items, schema)
        return constrain_fixed_tokens(
            masked_logits,
            schema.fixed_token_ids,
            schema.fixed_token_index,
        )

    if schema.state == GenerationState.PARAMETER_VAL:
        return constrain_para_val(masked_logits, vocab_items, schema)

    if schema.state == GenerationState.AFTER_PARAMETER_VAL:
        if schema.pending_after_val_text:
            return constrain_fixed_text(
                masked_logits,
                vocab_items,
                schema.pending_after_val_text,
            )

        if schema.fixed_token_index >= len(schema.fixed_token_ids):
            if schema.remaining_parameters:
                schema.state = GenerationState.PARAMETER_KEY
                schema.partial_para = ""
                return constrain_para_key(
                    masked_logits,
                    vocab_items,
                    schema,
                )

            schema.state = GenerationState.END
            return masked_logits

        return constrain_fixed_tokens(
            masked_logits,
            schema.fixed_token_ids,
            schema.fixed_token_index,
        )

    if schema.state == GenerationState.END:
        return masked_logits

    return masked_logits
