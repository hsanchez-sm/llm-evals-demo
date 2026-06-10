"""The EvalsInc assistant: ties together routing, retrieval and answer generation."""
from .retriever import TfidfRetriever
from .router import Router


class Assistant:
    def __init__(self, docs, entity_terms, backend, k=3):
        self.docs = docs
        self.retriever = TfidfRetriever(docs)
        self.router = Router(entity_terms)
        self.backend = backend
        self.k = k

    def retrieve(self, query, k=None):
        return self.retriever.retrieve(query, k or self.k)

    def answer(self, query, k=None):
        agent = self.router.route(query)
        hits = self.retrieve(query, k)
        doc_ids = [doc_id for doc_id, _ in hits]
        context = "\n\n".join(self.docs[i] for i in doc_ids)
        text = self.backend.generate(query, context)
        return {"agent": agent, "retrieved": doc_ids, "answer": text}
