import json, sys

# -------- Used to generate latex table from the json files --------
# Pass location and name of json file to output, like
# python gen_table.py outputs/cifar/small_mlp/classification_report.json

rep = json.load(open(sys.argv[1]))
for cls in map(str, range(10)):  # per-class rows
    p, r, f = rep[cls]["precision"], rep[cls]["recall"], rep[cls]["f1-score"]
    print(f"{cls} & {p:.3f} & {r:.3f} & {f:.3f} \\\\")
mac = rep["macro avg"]
acc = rep["accuracy"]
print(r"\midrule")
print(
    f"\\textbf{{Macro avg}} & {mac['precision']:.3f} & "
    f"{mac['recall']:.3f} & {mac['f1-score']:.3f} \\\\"
)
print(f"\\textbf{{Micro (=Acc.)}} & {acc:.3f} & {acc:.3f} & {acc:.3f} \\\\")
