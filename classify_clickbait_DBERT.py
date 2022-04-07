from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import json

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
model = AutoModelForSequenceClassification.from_pretrained("models/predict")
MAX_LENGTH = 20


# File to keep record of the classified files
classifiedFile = f"clasification/memory/classified_phrases_{MAX_LENGTH}.json"
classifiedPhrases = []
if os.path.exists(classifiedFile):

    classifiedPhrases = load_json_file(classifiedFile)

# File to keep record of all the labels for the phrases
labelsFile = f"clasification/results/labels_{MAX_LENGTH}.json"
labels = []
if os.path.exists(labelsFile):
    labels = load_json_file(labelsFile)


# Create the classifier
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Directory of the phrases
phrasesDirectory = f"output_phrases/{MAX_LENGTH}"
for file in os.listdir(phrasesDirectory):

    if file not in classifiedPhrases:
        file_location = phrasesDirectory+"/"+file
        sentences = load_json_file(file_location)

        # Analyze if the phrase is clickbait
        for sentence in sentences:
            result = classifier(sentence)
            label = 1
            if "0" in result["label"]:
                label = 0

            labels.append(label)

        # Write the results and that the file has been analyzed
        classifiedPhrases.append(file)

        write_json(classifiedFile, classifiedPhrases)
        write_json(labelsFile, labels)
