from hcs4os.classification_system.registry import get_classification_system
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def __reformat_code(code):
    if len(code) <= 4:
        return code
    else:
        return code[:4] + " " + code[4:]

def lookup(s):
    c = __reformat_code(s)
    return sea.get_code(c).description if c in sea._lookup else _to_add[c]

# Codes that are not in the official sea documentation need to be added. 
_to_add = {
    "1633 9":"Kombiversicherung für Kraftfahrzeug-Haftpflicht- und -Kaskoversicherung (wenn genaue Zuordnung nicht möglich)",
    '0980 1':"Pauschalreise im Inland (ohne Kurtaxe und Bettensteuer), auch Wochenend-, Städtereisen, Wall-, Ausflugsfahrten",
    '069':"Vorauszahlungen für pauschalierte Zuzahlungen zur gesetzlichen Krankenversicherung, z.B. bei chronischen Erkrankungen",
    '0980 2':"Pauschalreise ins Ausland (Europa und Fernreisen) auch Wochenend-, Städtereisen, Wall-, Ausflugsfahrten",
    '0321 4':"Schuhzubehör z.B. Schnürsenkel, Absätze, Einlegesohlen (ohne orthopädische Einlagen), Gamaschen, Schuhleisten",
    '1649 1':"Haushaltsbezogene geleistete Übertragungen",
    '0713 9':"Finanzierungsleasing von Fahrrädern, E-Bikes, Pedelecs und E-Scooter, Kauf nach Ablauf der Leasingdauer und Anzahlungsbetrag (ohne PKW und Kraft-, Jobräder)",
    '0711 9':"Finanzierungsleasing von Personenkraftwagen, mtl. Leasingrate, Kauf nach Ablauf der Leasingdauer und Anzahlungsbetrag (ohne Kraft-, Job-, Fahrräder, Golf-, Wohn-, Dienstwagen, -mobile, Versehrtenfahrzeuge, Reparatur, Wartung, Pflege)",
    '1612 2':"Kraftfahrzeugsteuer",
    '9999 998':"Sonstige Ausgaben - unspezifischer Konsum",
    '0442 9':"Sondermüllentsorgung (ohne Müllabfuhr), z.B. Gebühren an Wertstoffhöfe",
    '9999 991':"FA-Sondercode - bitte prüfen Sie die Umtragung in PFB/HFB.",
}

sea = get_classification_system("SEA_2021")

df = pd.read_parquet("data/training_data/raw_no_dupl2.parquet")[["klartext", "SEA5"]]
df["SEA"] = df["SEA5"].apply(lambda s: re.sub(r"0+$", "", s))
df["document"] = df["SEA"].apply(lookup)
uniq = sorted(set(df["klartext"].tolist()))
train_kt, temp_kt = train_test_split(uniq, test_size=0.2, random_state=42)
val_kt, test_kt = train_test_split(temp_kt, test_size=0.5, random_state=42)

train_kt_set = set(train_kt)
test_kt_set = set(test_kt)
val_kt_set = set(val_kt)


df_train = df[df.klartext.apply(lambda s: s in train_kt_set)].reset_index(drop=True)
df_test = df[df.klartext.apply(lambda s: s in test_kt_set)].reset_index(drop=True)
df_val = df[df.klartext.apply(lambda s: s in val_kt_set)].reset_index(drop=True)

print(len(f"Len: train{len(df_train)}"))

df_train.to_parquet("data/training_data/splits/train2.parquet")
df_test.to_parquet("data/training_data/splits/test2.parquet")
df_val.to_parquet("data/training_data/splits/val2.parquet")

