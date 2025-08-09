from src.metrics_pipeline import normalize_text

def test_trim_whitespace():
    assert normalize_text(" hello ") == "hello"

def test_preserve_characters():
    assert normalize_text("café") == "café"

def test_remove_extra_spaces():
    assert normalize_text("  spaced out  ") == "spaced out"

def test_empty_string():
    assert normalize_text("") == ""

def test_already_normalized():
    assert normalize_text("normalized") == "normalized"
