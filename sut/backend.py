"""Pluggable model backend.

- MockBackend: deterministic, dependency-free. Extractive answers/summaries built
  from the retrieved context. Lets the whole eval suite run in CI with no API key.
- AnthropicBackend / OpenAIBackend: real LLM calls (lazy-imported). Require an API
  key in the environment; used when you pass --backend anthropic|openai.

All backends expose the same interface:
    generate(query, context) -> str
    summarize(source, instruction) -> str
"""
import os

from .text_utils import sentences, tokenize


class MockBackend:
    name = "mock"

    def generate(self, query, context, n=2):
        """Extractive 'answer': the top-N context lines by query overlap.

        Value lines (those with a number or a 'field: value' colon) get a small
        bonus so a data line beats a matching header/title.
        """
        q = set(tokenize(query))
        scored = []
        for line in context.splitlines():
            s = line.strip()
            if not s or set(s) <= set("=-_ "):  # skip separators
                continue
            toks = set(tokenize(s))
            overlap = len(q & toks)
            if not toks or overlap == 0:
                continue
            bonus = 0.3 if (":" in s or any(c.isdigit() for c in s)) else 0.0
            scored.append((overlap + bonus, s))
        scored.sort(key=lambda x: -x[0])
        return " ".join(s for _, s in scored[:n])

    def summarize(self, source, instruction, n=4):
        """Extractive summary via greedy maximum-coverage of content tokens.

        Repeatedly picks the sentence that adds the most *new* information, which
        spreads coverage across distinct facts instead of clustering on one line.
        """
        sents = sentences(source)
        pool = list(enumerate(sents))
        chosen_idx, covered = [], set()
        while pool and len(chosen_idx) < n:
            best_i, best_gain, best_pos = None, 0, -1
            for pos, (i, s) in enumerate(pool):
                gain = len(set(tokenize(s)) - covered)
                if gain > best_gain:
                    best_i, best_gain, best_pos = i, gain, pos
            if best_i is None:
                break
            chosen_idx.append(best_i)
            covered |= set(tokenize(sents[best_i]))
            pool.pop(best_pos)
        keep = set(chosen_idx)
        return " ".join(s for i, s in enumerate(sents) if i in keep)


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model="claude-sonnet-4-5"):
        import anthropic  # lazy
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def _complete(self, system, user):
        msg = self._client.messages.create(
            model=self._model, max_tokens=400,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    def generate(self, query, context):
        return self._complete(
            "Answer the question using ONLY the context. If unknown, say so.",
            f"Context:\n{context}\n\nQuestion: {query}",
        )

    def summarize(self, source, instruction):
        return self._complete(
            "You summarize faithfully, inventing nothing.",
            f"{instruction}\n\nText:\n{source}",
        )


class OpenAIBackend:
    name = "openai"

    def __init__(self, model="gpt-4o-mini"):
        import openai  # lazy
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model

    def _complete(self, system, user):
        resp = self._client.chat.completions.create(
            model=self._model, max_tokens=400,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()

    def generate(self, query, context):
        return self._complete(
            "Answer the question using ONLY the context. If unknown, say so.",
            f"Context:\n{context}\n\nQuestion: {query}",
        )

    def summarize(self, source, instruction):
        return self._complete(
            "You summarize faithfully, inventing nothing.",
            f"{instruction}\n\nText:\n{source}",
        )


def get_backend(name):
    name = (name or "mock").lower()
    if name == "mock":
        return MockBackend()
    if name == "anthropic":
        return AnthropicBackend()
    if name == "openai":
        return OpenAIBackend()
    raise ValueError(f"Unknown backend: {name!r} (use mock | anthropic | openai)")
