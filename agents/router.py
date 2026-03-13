import re

from app.enums import Intent, Route
from constants import (
    MEMORY_INTENT_PHRASES,
    SMALLTALK_GREETING_FIRST,
    SMALLTALK_GREETING_PHRASES,
    SMALLTALK_THANKS_FIRST,
)

def router_agent(state):
    question = (state.get("question") or "").strip()
    normalized = " ".join(question.lower().split())

    if not normalized:
        return {
            "route": Route.SMALLTALK.value,
            "intent": Intent.SMALLTALK.value,
        }

    cleaned = re.sub(r"[^a-z0-9\s']+", "", normalized)
    tokens = [t for t in cleaned.split() if t]

    # Memory / conversation-history intent (should NOT run RAG)
    if cleaned in MEMORY_INTENT_PHRASES:
        return {"route": Route.MEMORY.value, "intent": Intent.MEMORY.value}

    is_smalltalk = False
    if cleaned in SMALLTALK_GREETING_PHRASES:
        is_smalltalk = True
    elif tokens and tokens[0] in SMALLTALK_GREETING_FIRST and len(tokens) <= 3:
        is_smalltalk = True
    elif tokens and tokens[0] in SMALLTALK_THANKS_FIRST and len(tokens) <= 4:
        is_smalltalk = True
    elif len(cleaned) <= 3:
        is_smalltalk = True

    if is_smalltalk:
        return {"route": Route.SMALLTALK.value, "intent": Intent.SMALLTALK.value}

    return {"route": Route.RESEARCH.value, "intent": Intent.RESEARCH.value}
