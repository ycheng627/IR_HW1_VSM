files = [
    "prediction-0.74784.csv",
    "prediction-0.75864.csv",
    "prediction-0.76710.csv",
    "prediction-0.75555.csv",
    "prediction-0.76362.csv",
    "prediction-0.77475.csv"
]

csv = []
for file_path in files:
    with open(file_path) as f:
        csv.append(f.read())

for f in csv:
    for line in f:
        print(line)

print(csv)