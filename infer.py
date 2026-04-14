import json
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from utils import merge_bio_tags


INTENT_MODEL_PATH = "models/intent_model"
SLOT_MODEL_PATH = "models/slot_model"


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def predict_intent(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        confidence = probs[0][pred_id].item()

    return model.config.id2label[pred_id], confidence


def predict_slots(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    labels = [model.config.id2label[p] for p in pred_ids]

    filtered_tokens = []
    filtered_labels = []
    for token, label in zip(tokens, labels):
        if token in tokenizer.all_special_tokens:
            continue
        filtered_tokens.append(token)
        filtered_labels.append(label)

    # simple wordpiece cleanup
    rebuilt_tokens = []
    rebuilt_labels = []

    for token, label in zip(filtered_tokens, filtered_labels):
        if token.startswith("##") and rebuilt_tokens:
            rebuilt_tokens[-1] += token[2:]
        else:
            rebuilt_tokens.append(token)
            rebuilt_labels.append(label)

    entities = merge_bio_tags(rebuilt_tokens, rebuilt_labels)
    return entities


def build_summary(intent, slots):
    slot_map = {}
    for k, v in slots:
        slot_map.setdefault(k, []).append(v)

    if intent == "report_issue":
        issue = slot_map.get("ISSUE_TYPE", [])
        obj = slot_map.get("OBJECT", [])
        if issue and obj:
            return f"report {issue[0]} {obj[0]}"
        if obj:
            return f"report {obj[0]}"
        return "report issue"

    if intent == "get_information":
        obj = slot_map.get("OBJECT", [])
        if obj:
            return f"get information on {obj[0]}"
        return "get information"

    if intent == "find_place":
        obj = slot_map.get("OBJECT", [])
        loc = slot_map.get("LOCATION", [])
        if obj and loc:
            return f"find {obj[0]} in {loc[0]}"
        if obj:
            return f"find {obj[0]}"
        return "find place"

    if intent == "update_profile":
        field = slot_map.get("FIELD", [])
        if field:
            return f"update {field[0]}"
        return "update profile"

    if intent == "place_order":
        return "place order"

    return intent


def main():
    intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
    intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)

    slot_tokenizer = AutoTokenizer.from_pretrained(SLOT_MODEL_PATH)
    slot_model = AutoModelForTokenClassification.from_pretrained(SLOT_MODEL_PATH)

    print("POC NLU ready. Type 'exit' to quit.")

    while True:
        text = input("Enter user message: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue

        text = clean_text(text)

        intent, confidence = predict_intent(text, intent_tokenizer, intent_model)
        slots = predict_slots(text, slot_tokenizer, slot_model)

        structured_slots = {}
        for key, value in slots:
            structured_slots.setdefault(key.lower(), []).append(value)

        output = {
            "utterance_type": "request" if intent not in {"provide_information", "other"} else intent,
            "intent": intent,
            "confidence": round(confidence, 4),
            "slots": structured_slots,
            "summary": build_summary(intent, slots),
            "input": text,
        }

        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()