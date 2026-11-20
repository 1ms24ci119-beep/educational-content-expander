"""Tokenization, n-gram language modelling, smoothing, and generation."""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.,!?;:]")

def tokenize(text: str) -> list[str]:
    """Split text into words and simple punctuation tokens."""
    return TOKEN_PATTERN.findall(text.lower())

def detokenize(tokens: Sequence[str]) -> str:
    """Join tokens into readable text with basic punctuation handling."""
    result = ""
    no_space_before = {".", ",", "!", "?", ";", ":"}
    for token in tokens:
        if token in no_space_before:
            result = result.rstrip() + token + " "
        else:
            result += token + " "
    return result.strip()

class NGramModel:
    """An n-gram model with add-k smoothing and weighted next-token sampling.

    ``n=2`` is a bigram model (one-word context); ``n=3`` is a trigram model.
    Unknown contexts back off to shorter contexts, eventually to unigram counts.
    """

    def __init__(self, n: int = 2, k: float = 0.5, seed: int | None = None):
        if n < 1:
            raise ValueError("n must be at least 1")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.n = n
        self.k = float(k)
        self.random = random.Random(seed)
        self.vocabulary: set[str] = set()
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.unigrams: Counter = Counter()
        self._trained = False

    def fit(self, texts: Iterable[str]) -> "NGramModel":
        """Train the model from an iterable of documents."""
        self.counts.clear()
        self.unigrams.clear()
        self.vocabulary.clear()
        for text in texts:
            tokens = tokenize(text)
            if not tokens:
                continue
            self.vocabulary.update(tokens)
            self.unigrams.update(tokens)
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for index in range(self.n - 1, len(padded)):
                context = tuple(padded[index - self.n + 1:index]) if self.n > 1 else ()
                target = padded[index]
                self.counts[context][target] += 1
        # Keep sentence termination available to the smoothed distribution.
        self.vocabulary.add("</s>")
        self._trained = bool(self.unigrams)
        if not self._trained:
            raise ValueError("The training corpus contains no usable words.")
        return self

    def _distribution(self, context: Sequence[str]) -> tuple[list[str], list[float]]:
        if not self._trained:
            raise RuntimeError("Train the model with fit() before generating text.")
        vocab = sorted(self.vocabulary)
        usable = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        # Back off until a context with observed continuations is found.
        while usable and usable not in self.counts:
            usable = usable[1:]
        observed = self.counts.get(usable, Counter())
        denominator = sum(observed.values()) + self.k * len(vocab)
        weights = [observed[word] + self.k for word in vocab]
        # Do not generate the start marker; it is not part of the vocabulary.
        return vocab, [weight / denominator for weight in weights]

    def predict_next(self, context: str | Sequence[str], top_n: int = 5) -> list[tuple[str, float]]:
        """Return the most likely next tokens and their smoothed probabilities."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        tokens = tokenize(context) if isinstance(context, str) else list(context)
        vocab, probabilities = self._distribution(tokens)
        ranked = sorted(zip(vocab, probabilities), key=lambda item: item[1], reverse=True)
        return ranked[:top_n]

    def generate(self, prompt: str, max_words: int = 25, temperature: float = 1.0) -> str:
        """Expand a prompt by sampling one token at a time."""
        if max_words < 1:
            raise ValueError("max_words must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if not self._trained:
            raise RuntimeError("Train the model with fit() before generating text.")
        generated = tokenize(prompt)
        initial_length = len(generated)
        for _ in range(max_words):
            vocab, probs = self._distribution(generated)
            adjusted = [p ** (1.0 / temperature) for p in probs]
            total = sum(adjusted)
            next_token = self.random.choices(vocab, weights=[p / total for p in adjusted], k=1)[0]
            if next_token == "</s>":
                # Avoid an empty expansion when a short prompt ends immediately.
                if len(generated) == initial_length:
                    continue
                break
            generated.append(next_token)
        return detokenize(generated)
