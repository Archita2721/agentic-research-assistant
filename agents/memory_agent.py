def memory_agent(state):
    history = state.get("chat_history", []) or []

    user_msgs = [m.get("content", "") for m in history if m.get("role") == "user" and m.get("content")]
    if not user_msgs:
        return {"final_answer": "I don't have any earlier messages for this session yet."}

    # Show the last few user questions (excluding the current one, if it matches).
    current = (state.get("question") or "").strip()
    cleaned = [m for m in user_msgs if m.strip() and m.strip() != current]
    recent = cleaned[-5:] if cleaned else user_msgs[-5:]

    lines = "\n".join(f"- {q}" for q in recent)
    return {"final_answer": f"Here’s what you asked earlier in this session:\n{lines}"}

