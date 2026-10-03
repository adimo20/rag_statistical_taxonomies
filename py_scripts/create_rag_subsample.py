import json
import random
from helpers import load_test_split
import sys

test_data = load_test_split()
random.Random(42).shuffle(test_data)
selected_data = test_data[:1000]
with open(sys.argv[1],"w", encoding="utf-8") as f:
    f.write(
        json.dumps(selected_data, indent=4, ensure_ascii=False)
    )

