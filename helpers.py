import json

def harmonize_code(code:str):
    return code if len(code) <= 4 else code[:4] + " " + code[4:]

def load_test_split():
    with open("data/training_data/splits/test2.json", "r", encoding="utf-8") as f:
        test_data = json.loads(f.read())
    return test_data


def load_val_split():
    with open("data/training_data/splits/val2.json", "r", encoding="utf-8") as f:
        val_data = json.loads(f.read())
    return val_data



def load_train_split():
    with open("data/training_data/splits/train2.json", "r", encoding="utf-8") as f:
        train_data = json.loads(f.read())
    return train_data



def get_sub_code_descriptions(code, cs):
    children = []
    to_explore = []

    curr = cs.get_children(code)
    children.extend(curr)
    to_explore += [c.code for c in curr]

    while to_explore!=[]:
        curr = cs.get_children(to_explore[0])
        if curr == []: 
            to_explore.pop(0)
            continue
        children.extend(curr)
        to_explore.extend([c.code for c in curr])
        to_explore.pop(0)

    return [c.description for c in children]

