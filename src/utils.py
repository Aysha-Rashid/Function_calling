import numpy as np
from typing import List
import re


def apply_token_mask(logits: np.ndarray, valid_ids: List[int]) -> np.ndarray:
    """Keep only the supplied token IDs."""
    if not valid_ids:
        raise ValueError("No valid tokens available")

    masked = np.full_like(logits, -float("inf"))
    masked[valid_ids] = logits[valid_ids]
    return masked


_NUMBER_COMPLETE_RE = re.compile(
    r"^-?(?:0|[1-9]\d*)"
    r"(?:\.\d+)?"
    r"(?:[eE][+-]?\d+)?$"
)


def is_number_prefix(value: str) -> bool:
    """Return True if valuecan still become a valid JSON number."""
    if value == "-":
        return True
    if not value:
        return False
    # Integer part
    match = re.match(r"^-?(0|[1-9]\d*)", value)
    if match is None:
        return False
    position = match.end()
    # Nothing else
    if position == len(value):
        return True
    # Decimal part
    if value[position] == ".":
        position += 1
        start_fraction = position
        while (position < len(value) and value[position].isdigit()):
            position += 1
        # "." or ".123" are valid prefixes
        if position == len(value):
            return True
        # Exponent is only allowed after at least
        # one fractional digit.
        if (position > start_fraction and value[position] in "eE"):
            position += 1
            if position == len(value):
                return True
            if value[position] in "+-":
                position += 1
                if position == len(value):
                    return True
            while (position < len(value) and value[position].isdigit()):
                position += 1
            return position == len(value)
        return False

    # Exponent without decimal part
    if value[position] in "eE":
        position += 1
        if position == len(value):
            return True
        if value[position] in "+-":
            position += 1
            if position == len(value):
                return True
        while (position < len(value) and value[position].isdigit()):
            position += 1
        return position == len(value)
    return False


def is_complete_number(value: str) -> bool:
    """Return True when valueis a complete JSON number."""
    return _NUMBER_COMPLETE_RE.fullmatch(value) is not None


_STRING_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


# Result codes returned by scan_string.
STRING_INCOMPLETE = "incomplete"   # still open, still a valid prefix
STRING_COMPLETE = "complete"       # terminated by a closing quote
STRING_INVALID = "invalid"         # can never become a valid JSON string


def scan_string(value: str) -> tuple[str, int]:
    """Classify *value* as the text of a JSON string.

    *value* is expected to start with the opening quote. The scan returns a
    ``(status, index)`` pair:

    * ``(STRING_INCOMPLETE, -1)`` -- no closing quote yet, but everything
      seen so far may still become a valid JSON string.
    * ``(STRING_COMPLETE, i)`` -- the string is terminated by the closing
      quote at index ``i``. ``value[:i + 1]`` is then the whole string and
      ``value[i + 1:]`` is text that follows it.
    * ``(STRING_INVALID, -1)`` -- no continuation can rescue *value*.

    Reporting the position of the closing quote, instead of only accepting a
    quote in the very last position, is what allows a token that fuses the
    closing quote with the JSON punctuation coming after the value. Such a
    token finishes the string legally even though its quote is not the last
    character of the token.
    """
    if not value or value[0] != '"':
        return STRING_INVALID, -1
    index = 1
    length = len(value)
    while index < length:
        char = value[index]
        if char == '"':
            return STRING_COMPLETE, index
        if char == "\\":
            if index + 1 >= length:
                # A trailing backslash may still become an escape sequence.
                return STRING_INCOMPLETE, -1
            escape = value[index + 1]
            if escape == "u":
                digits = value[index + 2:index + 6]
                if not all(digit in _HEX_DIGITS for digit in digits):
                    return STRING_INVALID, -1
                if len(digits) < 4:
                    # A partial escape sequence is still a valid prefix.
                    return STRING_INCOMPLETE, -1
                index += 6
                continue
            if escape not in _STRING_SIMPLE_ESCAPES:
                return STRING_INVALID, -1
            index += 2
            continue
        if ord(char) < 0x20:
            # Raw control characters are not allowed in JSON strings.
            return STRING_INVALID, -1
        index += 1
    # Unterminated, but everything so far is still a valid prefix.
    return STRING_INCOMPLETE, -1


def is_string_prefix(value: str) -> bool:
    """Return True if value can still become a valid JSON string.

    A string that is already closed counts as a valid prefix too: nothing
    has to be appended to it, but it must not be rejected either.
    """
    status, _ = scan_string(value)
    return status in (STRING_INCOMPLETE, STRING_COMPLETE)


def is_complete_string(value: str) -> bool:
    """Return True when value is exactly one complete JSON string."""
    status, close_index = scan_string(value)
    return status == STRING_COMPLETE and close_index == len(value) - 1


def closes_string(value: str, token_text: str) -> str | None:
    """Return the text that follows the string closed inside *token_text*.

    *value* is the string value generated so far and *token_text* is the
    decoded text of a candidate token. When appending the token terminates
    the string, the remainder of the token -- the part after the closing
    quote -- is returned so the caller can validate it against the
    punctuation the schema requires next. ``None`` means the token does not
    close the string.
    """
    status, close_index = scan_string(value + token_text)
    if status != STRING_COMPLETE:
        return None
    return token_text[close_index - len(value) + 1:]


_CONTROL_ESCAPES = {
    chr(code): "\\" + char
    for code, char in ((0x08, "b"), (0x09, "t"), (0x0A, "n"),
                       (0x0C, "f"), (0x0D, "r"))
}


def escape_string_control_chars(value: str) -> str:
    """Replace raw control characters inside a partial string by escapes.

    A model can emit a literal newline or tab where JSON requires the
    ``\\n`` / ``\\t`` escape. Such a value can never be completed as is,
    so the characters are rewritten to keep generation recoverable.
    """
    if not value:
        return value
    repaired = "".join(
        _CONTROL_ESCAPES.get(char, char) if ord(char) < 0x20 else char
        for char in value
    )
    return repaired


def is_complete_value(parameter_type: str, value: str) -> bool:
    """Return True when value is a complete JSON value of the given type."""
    if parameter_type == "number":
        return is_complete_number(value)
    if parameter_type == "boolean":
        return value in ("true", "false")
    if parameter_type == "string":
        return is_complete_string(value)
    raise ValueError(f"Unsupported parameter type: {parameter_type!r}")


def is_integer_prefix(value: str) -> bool:
    """Return True if valuecan still become a JSON integer."""

    if value == "-":
        return True
    if not value:
        return False
    return re.fullmatch(r"-?(0|[1-9]\d*)", value) is not None
