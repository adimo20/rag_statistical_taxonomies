import json
import fasttext

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
parser.add_argument("--path_train")
parser.add_argument("--path_test")
args = parser.parse_args()


OUTPUT_DIR = args.output_dir
PATH_TRAIN = args.path_train
PATH_TEST = args.path_test

# Fasttext takes its input data from a txt file in form of
# Free text 1 __label_1__\n
# Free text 2 __label_2__\n
with open(PATH_TRAIN, "r", encoding="utf-8") as f:
    train_data = json.loads(f.read())
train_formatted_ft = "\n".join([f"{ex["anchor"]} __label__{ex["label"]}" for ex in train_data])
with open(f"{OUTPUT_DIR}/train.txt", "w", encoding="utf-8") as f:
    f.write(train_formatted_ft)
    
with open(PATH_TEST, "r", encoding="utf-8") as f:
    test_data = json.loads(f.read())
test_formatted_ft = "\n".join([f"{ex["anchor"]} __label__{ex["label"]}" for ex in test_data])
with open(f"{OUTPUT_DIR}/test.txt", "w", encoding="utf-8") as f:
    f.write(test_formatted_ft)

model = fasttext.train_supervised(
    input=f"{OUTPUT_DIR}/train.txt",
    minn=2,
    maxn=4,
    wordNgrams=4
)

for n in [1, 3, 5, 10, 20]:
    print(n)
    print(model.test(f"{OUTPUT_DIR}/test.txt", k=n))

model.save_model(f"{OUTPUT_DIR}/model.bin")