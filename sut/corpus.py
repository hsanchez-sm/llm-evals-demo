"""Load the synthetic corpus, the ground-truth manifest and the JSONL datasets."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_manifest():
    return json.loads((ROOT / "data" / "corpus_manifest.json").read_text(encoding="utf-8"))


def read_doc(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def load_corpus():
    """Returns (docs, meta, manifest).

    docs: {doc_id: text} for the retrieval corpus (companies + employees + product).
    meta: {doc_id: manifest entry}.
    """
    manifest = load_manifest()
    docs = {d["doc_id"]: read_doc(d["path"]) for d in manifest["documents"]}
    meta = {d["doc_id"]: d for d in manifest["documents"]}
    return docs, meta, manifest


def source_texts(manifest):
    """All texts addressable by doc_id, including summarization-only transcripts."""
    entries = list(manifest["documents"]) + list(manifest.get("summarization_sources", []))
    return {d["doc_id"]: read_doc(d["path"]) for d in entries}


def entity_terms(meta, manifest):
    """Lowercased names the router uses to detect a knowledge-base entity in a query."""
    terms = set()
    for e in manifest["entities"].values():
        name = e["name"].lower()
        terms.add(name)
        terms.add(name.split()[0])  # distinctive first token, e.g. "nimbus"
    for d in meta.values():
        if d["type"] == "employee":
            person = d["title"].split("—")[0].strip().lower()
            if person:
                terms.add(person)
    return sorted(terms)


def load_jsonl(rel_path):
    rows = []
    for line in (ROOT / rel_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
