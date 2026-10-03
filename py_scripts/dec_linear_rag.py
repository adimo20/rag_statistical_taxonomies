import pandas as pd
import json
import re
 

def norm(code):
    return re.sub(r"[^0-9]", "", str(code)) if code is not None else None

 
def decompose_pipeline(eval_data):
    rows= []
    for key, case in eval_data.items():
        example = case["example"]
        gt = norm(example["sea_code"])
        candidates = [norm(c["code"]) for c in example["retrieved_candidates"]]
        top1 = candidates[0] if candidates else None
        response = case.get("response") or {}
        
        pred = norm(response.get("sea_code"))
        top1_correct = top1 == gt
        gt_in_candidates = gt in candidates
        pred_correct = pred == gt
        pred_rank = candidates.index(pred) + 1 if pred in candidates else None
        
        if pred is None:
            category = "generation error"
        elif top1_correct and pred_correct:
            category = "kept correct top-1"
        elif top1_correct and not pred_correct:
            category = "harmful switch"
        elif not top1_correct and pred_correct:
            category = "helpful switch"
        elif gt_in_candidates:
            category = "missed rescue"        
        else:
            category = "not retrievable"     

        rows.append({
            "id": key,
            "category": category,
            "top1_correct": top1_correct,
            "gt_in_candidates": gt_in_candidates,
            "pred_correct": pred_correct,
            "pred_rank": pred_rank,                     
            "pred_outside_candidates": pred is not None and pred not in candidates,

        })
    return pd.DataFrame(rows)


model_names = [
    'mistralai_Mistral-Small-4-119B-2603-NVFP4.json',
    'openai_gpt-oss-120b.json',
    'thinkingmachines_inkling-nvfp4.json',
    'google_gemma-4-31B-it.json',
    'mistralai_Mistral-Small-3.2-24B-Instruct-2506.json',
    'mistralai_Mistral-Large-3-675B-Instruct-2512.json'
]



settings = [
    'eval_linear_rag_subcodes',
    'eval_linear_rag_no_subcodes'   
]

settings_mapping = {
    'eval_linear_rag_subcodes':'w/ hierarchy',
    'eval_linear_rag_no_subcodes':'w/o hierarchy'   
}

BASE = "data"

eval_results = []

for setting in settings:
    for model_name in model_names:
        for k in [3,5,10]:
        
            with open(f"{BASE}/{setting}/{k}/{model_name}", "r", encoding="utf-8") as f:
                results = json.loads(f.read())

            _temp_res = decompose_pipeline(results)["category"].value_counts().to_dict()
            _temp_res["setting"] = settings_mapping[setting]
            _temp_res["k"] = k
            _temp_res["model_name"] = model_name.replace(".json", "").replace("_", "/")

            eval_results.append(_temp_res)

df_final = pd.DataFrame.from_dict(eval_results).sort_values(by=["model_name", "setting"])
df_final["complete_cases"] = df_final["generation error"].apply(lambda s: int(1000-s if not pd.isna(s) else 1000)) 
df_final.to_csv("decomposition.csv")
