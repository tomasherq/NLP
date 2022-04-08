from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import json

PHRASES_LOCATION = "../resources/click_bait/instances.jsonl"
RESULTS_LOCATION = "results/neural_net.py"
# Function to read JSON files


def load_json_file(filename):
    with open(filename, "r") as file_read:

        return json.load(file_read)

# Function to write JSON files


def write_json(filename, dict_to_print):
    with open(filename, "w") as file_write:

        return file_write.write(json.dumps(dict_to_print, indent=4))


# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("../models/predict")

sentences = {}
with open(PHRASES_LOCATION, "r") as file_read:

    for line in file_read:
        content = json.loads(line)
        sentences[content["id"]] = content["postText"]


labelsOriginal = {}
with open(PHRASES_LOCATION, "r") as file_read:

    for line in file_read:
        content = json.loads(line)
        labelsOriginal[content["id"]] = content["truthClass"] == "clickbait"

# Create the classifier
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

labels = {}
for sentenceID, sentenceContent in sentences.items():
    result = classifier(sentenceContent)
    label = True
    if "0" in result["label"]:
        label = False

    labels[sentenceID] = label

write_json("results/labels_neural_net.json", labels)

counterTotal = 0
counterCorrect = 0
for sentenceID in labels:
    if sentenceID in labelsOriginal:
        if labels[sentenceID] == labelsOriginal[sentenceID]:
            counterCorrect += 1
        counterTotal += 1

with open("results/neuralNet.txt", "w") as file_write:
    file_write.write(f"The total number of phrases is: {counterTotal}\n")
    file_write.write(f"The total number of correctly classified phrases is: {counterCorrect}\n")
    file_write.write(f"The total precision is: {round(counterCorrect/counterTotal,2)}\n")
