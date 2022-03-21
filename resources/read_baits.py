import json


labels = {}

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

with open("click_bait_phrases.json", "w") as file_write:

    file_write.write(json.dumps(clickbaitPhrases))
with open("click_bait_phrases.txt", "w") as file_write:

    for phrase in clickbaitPhrases:
        file_write.write(f'{phrase["text"]} {phrase["label"]}\n')
