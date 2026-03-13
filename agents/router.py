import re

from app.enums import Intent, Route

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

    greeting_first = {"hi", "hello", "hey", "yo", "sup", "wassup"}
    thanks_first = {"thanks", "thank", "thx"}
    greeting_phrases = {"whats up", "what's up", "thank you"}

    is_smalltalk = False
    if cleaned in greeting_phrases:
        is_smalltalk = True
    elif tokens and tokens[0] in greeting_first and len(tokens) <= 3:
        is_smalltalk = True
    elif tokens and tokens[0] in thanks_first and len(tokens) <= 4:
        is_smalltalk = True
    elif len(cleaned) <= 3:
        is_smalltalk = True

    if is_smalltalk:
        return {"route": Route.SMALLTALK.value, "intent": Intent.SMALLTALK.value}

    return {"route": Route.RESEARCH.value, "intent": Intent.RESEARCH.value}
