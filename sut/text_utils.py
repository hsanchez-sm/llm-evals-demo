"""Small, dependency-free text helpers shared by the retriever, router and graders."""
import re

# Minimal stopword list — kept small on purpose so we never drop domain keywords
# (e.g. "api", "plan", "refund") that the router and retriever rely on.
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "these", "those", "with", "by", "from", "as", "i", "you", "we", "they",
    "he", "she", "do", "does", "did", "my", "your", "our", "their", "there",
    "here", "what", "which", "who", "how", "me", "us",
}

# Capitalized words that are NOT proper nouns — excluded from "salient" extraction
# so a sentence-initial "The"/"This" isn't mistaken for an entity.
_CAP_STOP = {
    "The", "A", "An", "It", "This", "That", "These", "Those", "In", "On", "At",
    "Of", "For", "And", "Or", "But", "If", "As", "Is", "Are", "Was", "Were",
    "To", "With", "By", "From", "He", "She", "They", "We", "You", "I", "His",
    "Her", "Their", "Its", "How", "What", "Who", "Why", "When", "Where",
}

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")


def tokenize(text):
    """Lowercased content tokens, stopwords removed (for retrieval/keyword matching)."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def all_tokens(text):
    """Every lowercased token (stopwords kept) — used for presence checks."""
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def sentences(text):
    """Split into sentences / lines (handles transcripts and prose)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def salient_tokens(text):
    """Numbers and proper-noun-like tokens — the 'checkable facts' in a span.

    Used by the faithfulness grader: a summary token that is a number or a
    capitalized word (not a sentence filler) is something we can verify against
    the source.
    """
    out = []
    for m in re.findall(r"[A-Za-z]{2,}|\d+", text):
        if m.isdigit():
            out.append(m.lower())
        elif m[0].isupper() and m not in _CAP_STOP:
            out.append(m.lower())
    return out
