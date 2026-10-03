import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
import json
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

model = SentenceTransformer("models/embeddinggemma-300m")


train_df = pd.read_parquet("data/training_data/splits/train2.parquet")
test_df = pd.read_parquet("data/training_data/splits/test2.parquet")
val_df = pd.read_parquet("data/training_data/splits/val2.parquet")

train_df = train_df[["klartext","SEA", "document"]]
test_df = test_df[["klartext","SEA", "document"]]
val_df = val_df[["klartext","SEA", "document"]]
train_df.columns = ["anchor", "label", "document"]
test_df.columns = ["anchor", "label", "document"]
val_df.columns = ["anchor", "label", "document"]

train_pairs = Dataset.from_list(train_df[["anchor", "document"]].to_dict(orient="records"))
val_pairs = Dataset.from_list(val_df[["anchor", "document"]].to_dict(orient="records"))


train_ds = mine_hard_negatives(
    train_pairs,
    model,
    anchor_column_name="anchor",
    positive_column_name="document",
    num_negatives=5,
    range_min=0,          
    range_max=30,         
    sampling_strategy="top",
    output_format="triplet",
    batch_size=64,
    use_faiss=True, 
)

#Negative candidates mined, preparing dataset...
#Metric       Positive       Negative     Difference
#Count         513,865      2,569,325               
#Mean           0.2519         0.3872        -0.1353
#Median         0.2359         0.3799        -0.1369
#Std            0.1436         0.0873         0.1304
#Min           -0.1626         0.0735        -0.8020
#25%            0.1484         0.3281        -0.2210
#50%            0.2359         0.3799        -0.1369
#75%            0.3363         0.4340        -0.0545
#Max            0.9388         0.9148         0.5479

val_ds = mine_hard_negatives(
    val_pairs,
    model,
    anchor_column_name="anchor",
    positive_column_name="document",
    num_negatives=5,
    range_min=0,          
    range_max=30,         
    sampling_strategy="top",
    output_format="triplet",
    batch_size=64,
    use_faiss=True,     
)


#Negative candidates mined, preparing dataset...
#Metric       Positive       Negative     Difference
#Count          64,420        322,100               
#Mean           0.2522         0.3863        -0.1341
#Median         0.2365         0.3794        -0.1363
#Std            0.1433         0.0866         0.1307
#Min           -0.1464         0.0956        -0.7770
#25%            0.1489         0.3274        -0.2196
#50%            0.2365         0.3794        -0.1363
#75%            0.3375         0.4333        -0.0533
#Max            0.8584         0.8967         0.5191

train_ds.to_json("data/training_data/hard_negatives_mined/train_hn2.jsonl")
val_ds.to_json("data/training_data/hard_negatives_mined/val_hn2.jsonl")


val_json = val_df.to_dict(orient="records")
with open("data/training_data/splits/val2.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(val_json,indent=4, ensure_ascii=False))

train_json = train_df.to_dict(orient="records")
with open("data/training_data/splits/train2.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_json,indent=4, ensure_ascii=False))

test_json = test_df.to_dict(orient="records")
with open("data/training_data/splits/test2.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(test_json,indent=4, ensure_ascii=False))

