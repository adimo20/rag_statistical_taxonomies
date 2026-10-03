import json
from statistics import mean
from collections import Counter
import pandas as pd
import sys

MAX_ITERS = 5

BASE = sys.argv[1]

model_names = [
    "mistralai_Mistral-Small-4-119B-2603-NVFP4.json",
    "openai_gpt-oss-120b.json",
    "google_gemma-4-31B-it.json",
    "mistralai_Mistral-Small-3.2-24B-Instruct-2506.json",
    "mistralai_Mistral-Large-3-675B-Instruct-2512.json",
]

settings = [
    "agentic_rag_hierarchy_base_model", # zero-shot retriever w/ hierarchy
    "agentic_rag_no_hierarchy_base_model", # zero-shot retriever w/o hierarchy
    "eval_agentic_rag_hierarchy", # finetuned retriever w/ hierarchy
    "eval_agentic_rag_no_hierarchy", # finetuned retriever w/o hierarchy
]


def norm(code):
    return None if code is None else str(code).replace(" ", "").strip()


final_eval = []
for setting in settings:
    for model_name in model_names:
        with open(f"{BASE}/{setting}/{model_name}", "r", encoding="utf-8") as f:
            results = json.load(f)

        found_at_step = []      
        no_trajectory = 0       # runs crashed / returned no trajectory, most likely due to parser error or to poor capabilities of following the structured generation scheme.
        no_gold = 0             # examples without a gold label, this happens runs crash
        n_finished = 0          # agent called finish
        n_max_iters = 0         # never called finish, ran into the iteration limit
        n_other_stop = 0        # never called finish, fewer than MAX_ITERS steps, this is also an issue of following instruction, --> it doesnt call the finish tool build into dspy, but returns a reasoning, explanation and classification. No error at all.
        num_iterations = []
        num_codes_seen = []         # distinct num of codes the agent saw per run
        num_codes_seen_total = []   # number of all codes including repeats across searches

        for key, entry in results.items():
            gold = norm(entry["example"].get("sea_code"))
            if gold is None:
                no_gold += 1
                continue

            trajectory = entry["response"].get("trajectory")
            if trajectory is None:
                no_trajectory += 1
                continue

            tool_names = [v for s, v in trajectory.items() if s.startswith("tool_name")]
            called_finish = "finish" in tool_names

            num_iterations.append(sum(t != "finish" for t in tool_names))

            seen = [
                norm(c["code"])
                for s, obs in trajectory.items()
                if "observation" in s and isinstance(obs, list)
                for c in obs
            ]
            num_codes_seen.append(len(set(seen)))
            num_codes_seen_total.append(len(seen))

            observ_step = 1
            found = 0
            for step, obs in trajectory.items():
                if "observation" not in step:
                    continue
                if not isinstance(obs, list):  
                    continue
                codes = {norm(c["code"]) for c in obs}
                if gold in codes:
                    found = observ_step
                    break
                observ_step += 1

            if called_finish:
                n_finished += 1
            elif len(tool_names) >= MAX_ITERS:
                n_max_iters += 1
            else:
                n_other_stop += 1
            found_at_step.append(found)

        hits = [s for s in found_at_step if s != 0]
        final_eval.append({
            "model_name": model_name,
            "setting": setting,
            "mean_iter_correct_code_found": mean(hits) if hits else None,
            "num_correct_codes_found": len(hits),
            "num_evaluated": len(found_at_step),
            "retrieval_success": len(hits) / len(found_at_step) if found_at_step else None,
            "mean_number_iters": mean(num_iterations),
            "mean_codes_seen": mean(num_codes_seen),              
            "mean_codes_seen_total": mean(num_codes_seen_total),  
            "found_at_step":found_at_step
        })

df = pd.DataFrame(final_eval)
df["found_at_step_count"] = df.found_at_step.apply(lambda f: Counter(f))
df[["model_name", "setting", "found_at_step_count"]]
df.to_parquet("results.parquet")
