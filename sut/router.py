"""Rule-based router: classifies a user query to one of the four agents.

Order of precedence:
  1. Known KB entity mentioned  -> knowledge_agent  (handles the 'price of <company>' traps)
  2. Greeting / capability       -> general_agent
  3. Billing keywords            -> billing_agent
  4. Technical keywords          -> technical_support_agent
  5. Default                     -> knowledge_agent

This is intentionally simple and deterministic so it runs in CI without a model.
Swap in an LLM classifier via the real backend to compare against this baseline.
"""
from .text_utils import tokenize

BILLING_KW = {
    "price", "prices", "pricing", "cost", "costs", "plan", "plans", "charge",
    "charged", "charges", "refund", "refunds", "invoice", "invoiced", "billing",
    "subscription", "subscriptions", "upgrade", "downgrade", "pay", "payment", "money",
}
TECH_KW = {
    "error", "401", "403", "404", "429", "500", "503", "api", "password",
    "reset", "dashboard", "load", "loading", "key", "keys", "integration",
    "integrations", "connect", "2fa", "bug", "broken", "programmatic", "access",
    "endpoint", "token", "rate", "limit",
}
GENERAL_TOKENS = {"hi", "hello", "hey", "thanks", "thank", "thx", "bye", "goodbye"}
GENERAL_PHRASES = ("what can you do", "who are you", "what do you do", "how are you")

AGENTS = ("knowledge_agent", "billing_agent", "technical_support_agent", "general_agent")


class Router:
    def __init__(self, entity_terms):
        self.entity_terms = list(entity_terms)

    def route(self, query):
        q = query.lower()
        toks = set(tokenize(query))

        if any(term in q for term in self.entity_terms):
            return "knowledge_agent"
        if toks & GENERAL_TOKENS or any(p in q for p in GENERAL_PHRASES):
            return "general_agent"
        if toks & BILLING_KW:
            return "billing_agent"
        if toks & TECH_KW:
            return "technical_support_agent"
        return "knowledge_agent"
