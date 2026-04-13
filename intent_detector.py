import json
from transformers import pipeline


# Keep intents small and practical for a simple flow.
CANDIDATE_INTENTS = {
    "information on student visa": [
        "student visa",
        "apply for student visa",
        "student visa information",
    ],
    "greeting": [
        "hello",
        "hi",
        "good morning",
    ],
    "goodbye": [
        "bye",
        "goodbye",
        "see you",
    ],
    "thanks": [
        "thanks",
        "thank you",
        "appreciate it",
    ],
}


def build_label_map():
    """Create readable labels for zero-shot classification."""
    labels = []
    label_to_intent = {}
    for intent, examples in CANDIDATE_INTENTS.items():
        for ex in examples:
            label = f"{intent} :: {ex}"
            labels.append(label)
            label_to_intent[label] = intent
    return labels, label_to_intent


def detect_intent(message: str, classifier) -> dict:
    labels, label_to_intent = build_label_map()
    result = classifier(message, labels, multi_label=False)
    best_label = result["labels"][0]
    best_score = float(result["scores"][0])
    intent = label_to_intent[best_label]

    return {
        "intent": intent,
        "confidence": round(best_score, 4),
        "input": message,
    }


def main():
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )

    message = input("Enter user message: ").strip()
    output = detect_intent(message, classifier)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
