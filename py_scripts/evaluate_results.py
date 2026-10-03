import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import re
import sys

BASE = sys.argv[1]

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


final_eval = []

for setting in settings:
    for model_name in model_names:
        for k in [3, 5, 10]:


            with open(f"{BASE}/{setting}/{k}/{model_name}", "r", encoding="utf-8") as f:
                results = json.loads(f.read())


            freetexts = [results[str(i)]["example"]["input_expense"] for i in range(len(results))]
            ground_truth = [results[str(i)]["example"]["sea_code"] if "sea_code" in results[str(i)]["example"].keys() else "" for i in range(len(results))]

            prediction = [results[str(i)]["response"]["sea_code"] if "sea_code" in results[str(i)]["response"].keys() else "" for i in range(len(results))]
            
            ground_truth = [g for g, p in zip(ground_truth, prediction) if p is not None]
            prediction = [p for p in prediction if p is not None]
            
            ground_truth = [re.sub(r"[^0-9]", "", g) for g, p in zip(ground_truth, prediction) if p is not None]
            prediction = [re.sub(r"[^0-9]", "", p) for p in prediction if p is not None]
            

            reasoning =  [results[str(i)]["response"]["explaination"] if "explaination" in results[str(i)]["response"].keys() else "" for i in range(len(results))]

            final_eval.append(
                {
                    "precision_macro":round(precision_score(ground_truth, prediction, average='macro', zero_division=0), 4),
                    "precision_weighted":round(precision_score(ground_truth, prediction, average='weighted', zero_division=0), 4),
                    "recall_marcro":round(recall_score(ground_truth, prediction, average='macro', zero_division=0), 4),
                    "recall_weighted":round(recall_score(ground_truth, prediction, average='weighted', zero_division=0), 4),
                    "f1_score_macro":round(f1_score(ground_truth, prediction, average='macro', zero_division=0), 4),
                    "f1_score_weighted":round(f1_score(ground_truth, prediction, average='weighted', zero_division=0), 4),
                    "accuracy":round(accuracy_score(ground_truth, prediction), 4),
                    "k":k,
                    "model_name":model_name,
                    "setting":setting,
                    "number_of_results":len(prediction)
                }
            )
     
    
df_final_eval = pd.DataFrame.from_dict(final_eval)
df_final_eval["model_name"] = df_final_eval.model_name.apply(lambda s: s.replace(".json", "").replace("_", "/"))
df_final_eval["setting"] = df_final_eval.setting.apply(lambda s: settings_mapping[s])

df_final_eval.to_csv("linear_RAG.csv")

# ============================================================================================================================
# Agentic RAG
# ============================================================================================================================
model_names = [
    'mistralai_Mistral-Small-4-119B-2603-NVFP4.json',
    'openai_gpt-oss-120b.json',
    'google_gemma-4-31B-it.json',
    'mistralai_Mistral-Small-3.2-24B-Instruct-2506.json',
    'mistralai_Mistral-Large-3-675B-Instruct-2512.json'
]

settings = [
    "agentic_rag_hierarchy_base_model", # zero-shot retriever w/ hierarchy
    "agentic_rag_no_hierarchy_base_model", # zero-shot retriever w/o hierarchy
    "eval_agentic_rag_hierarchy", # finetuned retriever w/ hierarchy
    "eval_agentic_rag_no_hierarchy", # finetuned retriever w/o hierarchy
]


final_eval_agentic = []

for model_name in model_names:
    for setting in settings:
            with open(f"{BASE}/{setting}/{model_name}", "r", encoding="utf-8") as f:
                results = json.loads(f.read())


            freetexts = [results[str(i)]["example"]["input_expense"] for i in range(len(results))]
            ground_truth = [results[str(i)]["example"]["sea_code"] if "sea_code" in results[str(i)]["example"].keys() else None for i in range(len(results))]
            prediction = [results[str(i)]["response"]["sea_code"] if "sea_code" in results[str(i)]["response"].keys() else None for i in range(len(results))]
            
            ground_truth = [g for g, p in zip(ground_truth, prediction) if p is not None]
            prediction = [p for p in prediction if p is not None]
            
            ground_truth = [re.sub(r"[^0-9]", "", g) for g, p in zip(ground_truth, prediction) if p is not None]
            prediction = [re.sub(r"[^0-9]", "", p) for p in prediction if p is not None]
            

            reasoning =  [results[str(i)]["response"]["explaination"] if "explaination" in results[str(i)]["response"].keys() else "" for i in range(len(results))]

            final_eval_agentic.append(
                {
                    "precision_macro":round(precision_score(ground_truth, prediction, average='macro', zero_division=0), 4),
                    "precision_weighted":round(precision_score(ground_truth, prediction, average='weighted', zero_division=0), 4),
                    "recall_marcro":round(recall_score(ground_truth, prediction, average='macro', zero_division=0), 4),
                    "recall_weighted":round(recall_score(ground_truth, prediction, average='weighted', zero_division=0), 4),
                    "f1_score_macro":round(f1_score(ground_truth, prediction, average='macro', zero_division=0), 4),
                    "f1_score_weighted":round(f1_score(ground_truth, prediction, average='weighted', zero_division=0), 4),
                    "accuracy":round(accuracy_score(ground_truth, prediction), 4),
                    "model_name":model_name,
                    "setting":setting,
                    "number_of_results":len(prediction)
                }
            )


settings_mapping_2 = {
    'agentic_rag_hierarchy_base_model':"non-finetuned retriever w/ hierarchy",
    'agentic_rag_no_hierarchy_base_model':"non-finetuned retriever w/o hierarchy",
    'eval_agentic_rag_hierarchy':"finetuned retriever w/ hierarchy",
    'eval_agentic_rag_no_hierarchy':"finetuned retriever w/o hierarchy",   
}

 
    
df_final_eval_agentic = pd.DataFrame.from_dict(final_eval_agentic)
df_final_eval_agentic["model_name"] = df_final_eval_agentic.model_name.apply(lambda s: s.replace(".json", "").replace("_", "/"))
df_final_eval_agentic["setting"] = df_final_eval_agentic.setting.apply(lambda s: settings_mapping_2[s])

df_final_eval_agentic.to_csv("agentic_RAG.csv")

