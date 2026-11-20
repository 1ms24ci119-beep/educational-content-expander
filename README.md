# Educational Content Expander

A lightweight command-line tool that expands short educational phrases into
plausible continuations using a locally trained n-gram language model. It runs
offline and is intended as a learning project, not as a fact-checking system.

## Features

- Built-in corpora for general study skills, biology, computer science, and history.
- Bigram and trigram models.
- Add-k smoothing so unseen continuations do not receive zero probability.
- Back-off to shorter contexts when a longer n-gram has not been observed.
- Random sampling, configurable length and temperature, repeatable seeds.
- Custom text corpus support and saving output to a text file.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

Run from the project root:

```bash
python -m app.cli "photosynthesis"
python -m app.cli "binary search" --subject computer-science --n 3 --words 30
python -m app.cli "the circulatory system" --subject biology --words 20 --seed 7
python -m app.cli "the causes of" --corpus ./my_notes.txt --n 2 --output expanded.txt
```

Options:

| Option | Purpose |
|---|---|
| `--subject` | Select `general`, `biology`, `computer-science`, or `history` |
| `--corpus` | Use a custom UTF-8 text file instead of a bundled corpus |
| `--n` | `2` for bigram or `3` for trigram |
| `--k` | Add-k smoothing constant (must be greater than zero) |
| `--words` | Maximum number of generated words |
| `--temperature` | Sampling temperature; lower values favor common continuations |
| `--seed` | Set a random seed for repeatable output |
| `--output` | Save the result to a text file |

If no phrase is supplied, the CLI displays its help message.

## How it works

1. Each corpus line is treated as a short document and tokenized into words
   and punctuation.
2. The model counts n-gram contexts and their following tokens.
3. Add-k smoothing estimates a continuation probability as

   `P(word | context) = (count(context, word) + k) / (count(context) + kV)`

   where `V` is the vocabulary size.
4. When a trigram context is missing, the model backs off to a shorter context.
5. The generator samples a next token repeatedly until it reaches the word
   limit or a sentence-ending token.

## Milestones

- [x] Gather subject-relevant educational corpus samples.
- [x] Prepare corpus text and tokenize it for model training.
- [x] Build bigram/trigram n-gram counts.
- [x] Apply add-k smoothing and context back-off.
- [x] Generate continuations from short input phrases.
- [x] Provide a command-line interface.
- [x] Add subject and length controls.
- [x] Add optional output-file saving.
- [x] Add automated tests and CI workflow.

## Tests

```bash
python -m pytest -q
```

The bundled corpora are deliberately small examples. For better fluency and
subject relevance, supply a larger, clean, appropriately licensed educational
corpus using `--corpus`. N-gram output is statistical and may be repetitive or
factually incorrect; educators should review generated text before use.

## Git development history

The repository includes a sequence of eight focused commits documenting the
implementation milestones.
