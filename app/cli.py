"""Command-line interface for expanding educational notes."""
import argparse
import sys

from .core import NGramModel
from .storage import available_subjects, load_corpus, save_expansion

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edu-expand",
        description="Expand short educational notes using a smoothed n-gram model.",
    )
    parser.add_argument("text", nargs="?", help="short phrase or bullet point to expand")
    parser.add_argument("--subject", choices=available_subjects(), default="general",
                        help="bundled educational subject corpus")
    parser.add_argument("--corpus", help="path to a custom UTF-8 text corpus")
    parser.add_argument("--n", type=int, choices=(2, 3), default=2,
                        help="model order: 2=bigram, 3=trigram (default: 2)")
    parser.add_argument("--k", type=float, default=0.5, help="add-k smoothing value (default: 0.5)")
    parser.add_argument("--words", type=int, default=25, help="maximum new words to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--seed", type=int, help="random seed for repeatable output")
    parser.add_argument("--output", help="optional path to save the expansion")
    return parser

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.text:
        parser.print_help()
        return 0
    try:
        corpus = load_corpus(args.subject, args.corpus)
        model = NGramModel(n=args.n, k=args.k, seed=args.seed).fit(corpus)
        result = model.generate(args.text, max_words=args.words, temperature=args.temperature)
        print(result)
        if args.output:
            saved = save_expansion(result, args.output)
            print(f"Saved to: {saved}")
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
