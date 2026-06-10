"""CLI entry point: run one or all eval suites against a chosen backend.

Examples:
    python -m evals.run                      # all suites, mock backend
    python -m evals.run --suite routing
    python -m evals.run --backend anthropic  # needs ANTHROPIC_API_KEY
"""
import argparse
import sys

from sut.corpus import load_corpus, entity_terms, source_texts, load_jsonl
from sut.assistant import Assistant
from sut.backend import get_backend

from . import routing, retrieval, response_quality, summarization
from .report import save_and_print

SUITES = ("routing", "retrieval", "quality", "summarization")

DATASETS = {
    "routing": "data/datasets/agent_routing.jsonl",
    "retrieval": "data/datasets/rag_retrieval.jsonl",
    "quality": "data/datasets/response_quality.jsonl",
    "summarization": "data/datasets/summarization.jsonl",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run EvalsInc LLM eval suites.")
    parser.add_argument("--backend", default="mock", choices=["mock", "anthropic", "openai"])
    parser.add_argument("--suite", default="all", choices=["all", *SUITES])
    parser.add_argument("--k", type=int, default=3, help="top-k for retrieval")
    args = parser.parse_args(argv)

    docs, meta, manifest = load_corpus()
    backend = get_backend(args.backend)
    assistant = Assistant(docs, entity_terms(meta, manifest), backend, k=args.k)

    selected = SUITES if args.suite == "all" else (args.suite,)
    results = []

    if "routing" in selected:
        results.append(routing.run(assistant, load_jsonl(DATASETS["routing"])))
    if "retrieval" in selected:
        results.append(retrieval.run(assistant, load_jsonl(DATASETS["retrieval"]), k=args.k))
    if "quality" in selected:
        results.append(response_quality.run(assistant, load_jsonl(DATASETS["quality"])))
    if "summarization" in selected:
        results.append(summarization.run(
            assistant, load_jsonl(DATASETS["summarization"]), source_texts(manifest)))

    save_and_print(results, backend.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
