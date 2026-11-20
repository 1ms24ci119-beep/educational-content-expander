"""Loading educational corpora and saving generated expansions."""
from pathlib import Path

CORPUS_DIR = Path(__file__).with_name("corpora")

SUBJECT_FILES = {
    "biology": "biology.txt",
    "computer-science": "computer_science.txt",
    "history": "history.txt",
    "general": "general.txt",
}

def available_subjects() -> list[str]:
    return sorted(SUBJECT_FILES)

def load_corpus(subject: str = "general", corpus_path: str | None = None) -> list[str]:
    """Load one bundled subject corpus or a user-provided UTF-8 text file."""
    path = Path(corpus_path).expanduser() if corpus_path else CORPUS_DIR / SUBJECT_FILES.get(subject, "")
    if not path.is_file():
        choices = ", ".join(available_subjects())
        raise FileNotFoundError(f"Corpus not found for '{subject}'. Available subjects: {choices}")
    text = path.read_text(encoding="utf-8")
    documents = [part.strip() for part in text.splitlines() if part.strip()]
    if not documents:
        raise ValueError(f"Corpus file is empty: {path}")
    return documents

def save_expansion(text: str, output_path: str) -> Path:
    """Save generated text, creating parent folders when necessary."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.rstrip() + "\n", encoding="utf-8")
    return destination
