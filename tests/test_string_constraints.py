
from src.algo import (
    _string_fragment_status,
    constrain_string_value,
    is_valid_json_string_content,
)
from src.schema import DecodingStateJson


class FakeModel:
    def __init__(self, pieces):
        self.pieces = pieces

    def encode(self, text):
        return [[self.pieces.index(text)]]

    def decode(self, ids):
        return "".join(self.pieces[token_id] for token_id in ids)


def _valid_ids(pieces, current=""):
    model = FakeModel(pieces)
    state = DecodingStateJson()
    state.string_started = True
    state.parameter_value_generated = current
    state.string_content_token_ids = [pieces.index('"')]
    if current:
        state.string_content_token_ids.append(pieces.index(current))
    vocab = {i: f"vocab spelling {i}" for i in range(len(pieces))}
    logits = [1.0] * len(pieces)
    masked = constrain_string_value(logits, vocab, state, model)
    return {i for i, value in enumerate(masked) if value != float("-inf")}


def test_json_string_prefix_validation():
    for value in (
        "plain words, punctuation!",
        r'a \"quoted\" and \\ path',
        r"line1\nline2\ttab",
        "",
        "ends with ,}",
    ):
        assert is_valid_json_string_content(value) == (True, True)

    assert is_valid_json_string_content('unescaped " quote')[0] is False
    assert is_valid_json_string_content("raw\nnewline")[0] is False
    assert is_valid_json_string_content("unfinished\\")[0:2] == (True, False)
    assert is_valid_json_string_content(r"\u12")[0:2] == (True, False)


def test_string_tokens_use_decoded_text_and_allow_escaped_quotes():
    pieces = ['"', "hello ", r'\"', "\\", "x", ",", "}", "\n", "\t"]
    allowed = _valid_ids(pieces, "hello ")
    assert 2 in allowed  # escaped quote
    assert 3 in allowed  # escaped backslash
    assert 5 in allowed  # commas are ordinary content
    assert 6 in allowed  # braces are ordinary content
    assert 0 in allowed  # closing quote


def test_incomplete_escape_does_not_allow_close():
    pieces = ['"', "line\\", "n", "newline"]
    allowed = _valid_ids(pieces, "line\\")
    # A quote after a backslash completes the valid \\" escape; a second
    # quote is then needed to close the JSON string.
    assert is_valid_json_string_content('line\\"') == (True, True)
    model = FakeModel(pieces)
    assert _string_fragment_status(model.decode([0, 1, 0]))[:2] == (True, False)
    assert _string_fragment_status(model.decode([0, 1, 0, 0]))[:2] == (True, True)
    assert 2 in allowed


def test_string_close_allows_only_object_delimiter():
    pieces = [",", "}", '"', "x"]
    model = FakeModel(pieces)
    state = DecodingStateJson()
    state.parameter_key = ["first", "second"]
    state.parameter_index = 0
    state.string_started = True
    state.string_closed = True
    state.string_content_token_ids = [2, 2]
    masked = constrain_string_value([1.0] * len(pieces), {i: "ignored" for i in range(4)}, state, model)
    assert [i for i, value in enumerate(masked) if value != float("-inf")] == [0]

    state.parameter_index = 1
    masked = constrain_string_value([1.0] * len(pieces), {i: "ignored" for i in range(4)}, state, model)
    assert [i for i, value in enumerate(masked) if value != float("-inf")] == [1]
