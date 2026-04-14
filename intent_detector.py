import json
import os
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = os.getenv("INTENT_MODEL", "google/flan-t5-base")


FILLER_PREFIXES = [
    "hello",
    "hi",
    "hey",
    "please",
    "can you",
    "could you",
    "i want to",
    "i need to",
    "i would like to",
    "help me",
    "i need help",
    "i have to",
]

COMMON_TYPO_FIXES = {
    "porthole": "pothole",
    "grabage": "garbage",
    "flicekring": "flickering",
    "restaruant": "restaurant",
}


def _fallback_intent(message: str) -> str:
    text = re.sub(r"\s+", " ", message.strip().lower())
    patterns = [
        r"(?:i need to|i want to|can you|please)\s+(.+)",
        r"(?:help me|need help)\s+(?:to|with)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(" .!?")
    return text.strip(" .!?") or "unknown"


def _normalize_intent(intent: str) -> str:
    text = re.sub(r"\s+", " ", intent.strip().lower())
    for wrong, fixed in COMMON_TYPO_FIXES.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", fixed, text)

    # Remove leading filler phrases.
    changed = True
    while changed:
        changed = False
        for prefix in FILLER_PREFIXES:
            if text.startswith(prefix + " "):
                text = text[len(prefix):].strip()
                changed = True

    text = re.sub(r"^(to|that)\s+", "", text).strip()
    return text.strip(" .!?") or "unknown"


def detect_intent(message: str, tokenizer, model) -> dict:
    prompt = (
        "Extract a short problem-intent phrase from the user message.\n"
        "Rules:\n"
        "1) Return only the intent phrase (no JSON, no explanation).\n"
        "2) Keep it short: 2 to 8 words.\n"
        "3) Focus on user need/action.\n"
        "4) Fix obvious spelling mistakes.\n\n"
        "Examples:\n"
        "message: I need to place an order for 5 person in my home\n"
        "intent: order for 5 person\n"
        "message: hello i need to file a complaint for missing garbage pickup\n"
        "intent: file complaint for missing garbage pickup\n"
        "message: hi can i get information on student visa process\n"
        "intent: information on student visa process\n\n"
        f"message: {message}\n"
        "intent:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if generated.lower().startswith("intent:"):
            generated = generated.split(":", 1)[1].strip()
        generated = re.sub(r"\s+", " ", generated).strip(" .!?").lower()
        intent = _normalize_intent(generated) if generated else _fallback_intent(message)
    except Exception:
        intent = _fallback_intent(message)

    intent = _normalize_intent(intent)
    return {"intent": intent, "input": message}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print(f"Intent model loaded: {MODEL_NAME}")
    print("Type 'exit' to quit.")

    while True:
        message = input("Enter user message: ").strip()
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break
        output = detect_intent(message, tokenizer, model)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
