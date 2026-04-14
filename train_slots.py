import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import numpy as np
from seqeval.metrics import f1_score, accuracy_score


MODEL_NAME = "distilbert-base-uncased"

LABELS = [
    "O",
    "B-ACTION", "I-ACTION",
    "B-OBJECT", "I-OBJECT",
    "B-ISSUE_TYPE", "I-ISSUE_TYPE",
    "B-LOCATION", "I-LOCATION",
    "B-FIELD", "I-FIELD",
    "B-VALUE", "I-VALUE",
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def align_labels_with_tokens(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
    )

    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            else:
                raw_label = labels[word_idx]
                if raw_label.startswith("B-"):
                    raw_label = "I-" + raw_label[2:]
                label_ids.append(label2id.get(raw_label, label2id[labels[word_idx]]))
            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        current_preds = []
        current_labels = []
        for p_i, l_i in zip(prediction, label):
            if l_i != -100:
                current_preds.append(id2label[p_i])
                current_labels.append(id2label[l_i])
        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    return {
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }


def main():
    train_rows = load_jsonl("data/slots_train.jsonl")
    val_rows = load_jsonl("data/slots_val.jsonl")

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(lambda x: align_labels_with_tokens(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: align_labels_with_tokens(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="models/slot_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=7,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("models/slot_model")
    tokenizer.save_pretrained("models/slot_model")


if __name__ == "__main__":
    main()