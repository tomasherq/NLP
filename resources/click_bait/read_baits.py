import json

labels = {}

# Read from one dataset and get the indices of the phrases
# that are classified as clickbaits.
with open("truth.jsonl", "r") as file_read:

    for line in file_read:
        content = json.loads(line)
        labels[content['id']] = content['truthClass'] == "clickbait"


# Read from another dataset and get the indices texts
# of the phrases that are classified as clickbait.
clickbaitPhrases = list()
with open("instances.jsonl", "r") as file_read:

    for line in file_read:
        content = json.loads(line)

        if content['id'] in labels:

            clickbaitPhrases.append({"text": content['postText'][0], "label": labels[content["id"]]})

# Store the clickbait phrases in a JSON to tune the text generation model
with open("click_bait_phrases.json", "w") as file_write:
    file_write.write(json.dumps(clickbaitPhrases, indent=4))
