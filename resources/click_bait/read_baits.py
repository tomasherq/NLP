import json
import csv
from textwrap import indent

labels = {}

# Read from one dataset

with open("truth.jsonl", "r") as file_read:

    for line in file_read:
        content = json.loads(line)
        labels[content['id']] = content['truthClass'] == "clickbait"

clickbaitPhrases = list()

with open("instances.jsonl", "r") as file_read:

    for line in file_read:
        content = json.loads(line)

        if content['id'] in labels:

            clickbaitPhrases.append({"text": content['postText'][0], "label": labels[content["id"]]})

with open("clickbait_data.csv", "r") as file_read:
    readercsv = csv.reader(file_read, delimiter=',')
    for row in readercsv:
        clickbaitPhrases.append({"text": row[0], "label": row[1] == 1})

with open("click_bait_phrases.json", "w") as file_write:
    file_write.write(json.dumps(clickbaitPhrases, indent=4))
