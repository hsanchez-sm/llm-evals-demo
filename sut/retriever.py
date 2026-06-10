"""A small, deterministic TF-IDF retriever (pure stdlib, no external deps)."""
import math
from collections import Counter

from .text_utils import tokenize


class TfidfRetriever:
    def __init__(self, docs):
        # docs: {doc_id: text}
        self.ids = list(docs)
        self._tokens = {i: tokenize(t) for i, t in docs.items()}
        self.N = len(self.ids)
        self._df = Counter()
        for toks in self._tokens.values():
            for t in set(toks):
                self._df[t] += 1
        self._vectors = {i: self._vectorize(toks) for i, toks in self._tokens.items()}

    def _idf(self, term):
        return math.log((self.N + 1) / (self._df.get(term, 0) + 1)) + 1.0

    def _vectorize(self, toks):
        if not toks:
            return {}
        tf = Counter(toks)
        n = len(toks)
        return {t: (c / n) * self._idf(t) for t, c in tf.items()}

    @staticmethod
    def _cosine(a, b):
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b[t] for t in a if t in b)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def retrieve(self, query, k=5):
        """Return [(doc_id, score), ...] sorted by descending similarity."""
        qv = self._vectorize(tokenize(query))
        scored = [(i, self._cosine(qv, self._vectors[i])) for i in self.ids]
        # stable, deterministic ordering: score desc, then doc_id asc
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
