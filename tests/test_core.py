import pytest

from app.core import NGramModel, detokenize, tokenize

def test_tokenize_and_detokenize():
    tokens = tokenize("Cells are useful!")
    assert tokens == ["cells", "are", "useful", "!"]
    assert detokenize(tokens) == "cells are useful!"

def test_bigram_predicts_observed_continuation():
    model = NGramModel(n=2, k=0.1, seed=1).fit(["plants absorb light", "plants absorb energy"])
    predictions = dict(model.predict_next("plants", top_n=10))
    assert predictions["absorb"] > 0

def test_add_k_gives_unseen_words_nonzero_probability():
    model = NGramModel(n=2, k=1.0).fit(["water moves through cells"])
    predictions = dict(model.predict_next("water", top_n=20))
    assert predictions["cells"] > 0

def test_generate_keeps_prompt_and_adds_text():
    model = NGramModel(n=2, seed=4).fit([
        "photosynthesis converts light energy into chemical energy",
        "plants use light energy to make glucose",
    ])
    result = model.generate("photosynthesis", max_words=8)
    assert result.startswith("photosynthesis")
    assert len(tokenize(result)) > 1

def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        NGramModel(n=0)
    with pytest.raises(ValueError):
        NGramModel(k=0)
