from csv import DictReader, DictWriter

content = []
with open("../data/traintest.csv", "r", encoding="utf8") as file:
    reader = DictReader(file)
    for row in reader:
        content.append({"text": row["text"], "intent": row["intent"]})

with open("../data/traintest.csv", "w") as file:
    writer = DictWriter(file, fieldnames=["text", "intent"], quoting=1)
    # writer.writeheader()
    writer.writerows(content)
