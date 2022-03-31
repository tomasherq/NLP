from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import json


def load_json_file(filename):
    with open(filename, "r") as file_read:

        return json.load(file_read)


def write_json(filename, dict_to_print):
    with open(filename, "w") as file_write:

        return file_write.write(json.dumps(dict_to_print, indent=4))


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("models/predict")
MAX_LENGTH = 20

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

classifiedFile = f"clasification/memory/classified_phrases_{MAX_LENGTH}.json"
classifiedPhrases = []
if os.path.exists(classifiedFile):

    classifiedPhrases = load_json_file(classifiedFile)


labelsFile = f"clasification/results/labels_{MAX_LENGTH}.json"
labels = []
if os.path.exists(labelsFile):

    labels = load_json_file(labelsFile)

phrasesDirectory = f"output_phrases/{MAX_LENGTH}"
for file in os.listdir(phrasesDirectory):

    if file not in classifiedPhrases:
        file_location = phrasesDirectory+"/"+file
        sentences = load_json_file(file_location)
        for sentence in sentences:
            result = classifier("Meet the happiest #dog in the world!")
            label = 1
            if "0" in result["label"]:
                label = 0

            labels.append(label)

        classifiedPhrases.append(file)

        write_json(classifiedFile, classifiedPhrases)
        write_json(labelsFile, labels)
