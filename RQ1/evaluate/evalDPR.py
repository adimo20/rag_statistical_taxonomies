from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import argparse
import json
import os
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--test-file")
parser.add_argument("--experiment")
parser.add_argument("--run-name")
parser.add_argument("--version")
parser.add_argument("--tracking-uri")
parser.add_argument("--out_dir")
args = parser.parse_args()

with open(args.test_file, "r", encoding="utf-8") as f:
    test_ir_corpus = json.loads(f.read())


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


def sanitize_metric_name(name):
    # MLflow has problems processing @ so it will be replaced with _at_
    return name.replace("@", "_at_")


queries, corpus, relevant_docs = prepare_ir_corpus(test_ir_corpus)

k_values = {
    "mrr_at_k": [10],
    "ndcg_at_k": [10],
    "accuracy_at_k": [1, 3, 5, 10, 20],
    "precision_recall_at_k": [1, 3, 5, 10, 20],
    "map_at_k": [10],
}

model = SentenceTransformer(args.model, trust_remote_code=True)

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="dev-ir",
    show_progress_bar=True,
    **k_values,
)

mlflow.set_tracking_uri(args.tracking_uri)
mlflow.set_experiment(args.experiment)

with mlflow.start_run(run_name=args.run_name):
    
    mlflow.log_params(
        {
            "model": args.model,
            "test_file": args.test_file,
            "num_queries": len(queries),
            "num_corpus_docs": len(corpus),
            "evaluator_name": ir_evaluator.name,
            **{k: str(v) for k, v in k_values.items()},
        }
    )
    mlflow.set_tag("task", "information_retrieval")

    results = ir_evaluator(model)

    metrics = {
        sanitize_metric_name(name): value
        for name, value in results.items()
        if isinstance(value, (int, float))
    }
    mlflow.log_metrics(metrics)

    with open(f"{args.output_dir}/{args.version}_ir_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(results)