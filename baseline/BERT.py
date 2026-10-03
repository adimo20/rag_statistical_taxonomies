import json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--output_dir")
parser.add_argument("--path_train")
parser.add_argument("--path_test")
args = parser.parse_args()


MODEL_NAME =args.model_name
OUTPUT_DIR = args.output_dir
PATH_TRAIN = args.path_train
PATH_TEST = args.path_test
SEED=42

# This script has been used for the training of gelectra and german-bert-base-uncased via cli.

with open(PATH_TRAIN, "r", encoding="utf-8") as f:
    train_data = json.loads(f.read())

with open(PATH_TEST, "r", encoding="utf-8") as f:
    test_data = json.loads(f.read())

labels = sorted({ex["label"] for ex in train_data})
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(labels)


with open(f"{OUTPUT_DIR}/label2id.json", "w", encoding="utf-8") as f:
    f.write(
        json.dumps(label2id, indent=4, ensure_ascii=False)
    )

with open(f"{OUTPUT_DIR}/id2label.json", "w", encoding="utf-8") as f:
    f.write(
        json.dumps(id2label, indent=4, ensure_ascii=False)
    )


def to_dataset(data):
    return Dataset.from_dict({
        "text": [ex["anchor"] for ex in data],
        "label": [label2id[ex["label"]] for ex in data],
    })

train_ds = to_dataset(train_data)
test_ds = to_dataset([ex for ex in test_data if ex["label"] in label2id])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, gold = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    ranked = np.argsort(-logits, axis=1)  
    metrics = {}
    for n in [1, 3, 5, 10, 20]:
        if n > logits.shape[1]:
            continue
        topk = ranked[:, :n]
        hits = np.any(topk == gold[:, None], axis=1)
        metrics[f"top{n}_acc"] = float(hits.mean())
    return metrics

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="top1_acc",
    warmup_ratio=0.1,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    seed=SEED,
    data_seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

print(trainer.evaluate())

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)