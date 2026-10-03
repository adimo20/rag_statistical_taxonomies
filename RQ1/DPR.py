from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import json
import mlflow
from datasets import Dataset
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--run_name")
parser.add_argument("--mlflow_tracking_uri")
parser.add_argument("--mlflow_experiment_name")
parser.add_argument("--path_train_set")
parser.add_argument("--val_set_path")
args = parser.parse_args()

model_name = args.model_name
model_output_dir = f"/posit_share/home/montag-a/Adrian_Masterarbeit/models/Bi-Encoder/{args.run_name}"
save_pretrained_path = f"/posit_share/home/montag-a/Adrian_Masterarbeit/models/Bi-Encoder/{args.run_name}/final/"
mlflow_tracking_uri = args.mlflow_tracking_uri
mlflow_experiment_name = args.mlflow_experiment_name
path_train_set = args.path_train_set
val_set_path = args.val_set_path

print(model_name)
print(model_output_dir)
print(save_pretrained_path)
print(mlflow_tracking_uri)
print(mlflow_experiment_name)
print(path_train_set)
print(val_set_path)

train_ds = Dataset.from_json(path_train_set)

with open(val_set_path, "r", encoding="utf-8") as f:
    val_ir_corpus = json.loads(f.read())


def prepare_ir_corpus(ds):
    queries, corpus, relevant_docs = {}, {}, {}
    text_to_cid = {}
    for idx, sample in enumerate(ds):
        qid = f"q{idx}"
        queries[qid] = sample["anchor"]
        pos = sample["document"]
        if pos not in text_to_cid:
            text_to_cid[pos] = f"d{len(text_to_cid)}"
        cid = text_to_cid[pos]
        corpus[cid] = pos
        relevant_docs[qid] = {cid}
    return queries, corpus, relevant_docs


queries, corpus, relevant_docs = prepare_ir_corpus(val_ir_corpus)

eval_ds = Dataset.from_list(
    [{"anchor": s["anchor"], "document": s["document"]} for s in val_ir_corpus]
)

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment_name)

model = SentenceTransformer(model_name, trust_remote_code=True)

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="dev-ir",
    show_progress_bar=True,
    mrr_at_k=[10],
    ndcg_at_k=[10],
    accuracy_at_k=[1, 3, 5, 10, 20],
    precision_recall_at_k=[1, 3, 5, 10, 20],
    map_at_k=[10],
)

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir=model_output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32, 
    learning_rate=2e-5,
    warmup_ratio=0.1,                          
    batch_sampler=BatchSamplers.BATCH_SAMPLER,
    fp16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=10000,
    save_strategy="steps",
    save_steps=10000,
    save_total_limit=4,
    report_to=["mlflow"],
    seed=42,
    data_seed=42,
)

with mlflow.start_run():
    mlflow.log_params({
        "base_model": model_name,
        "train_set": path_train_set,
        "loss": type(loss).__name__,
        "train_samples": len(train_ds),
        "eval_queries": len(queries),
        "eval_corpus_size": len(corpus),
    })

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds, 
        evaluator=ir_evaluator,   
        loss=loss,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        final_metrics = ir_evaluator(model)
        clean_metrics = {
            k.replace("@", "_at_"): v
            for k, v in final_metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(clean_metrics)
        model.save_pretrained(save_pretrained_path)