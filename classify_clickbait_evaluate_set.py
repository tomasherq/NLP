from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import csv
from sklearn.metrics import confusion_matrix
import time
import pandas as pd

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("models/predict")

# Create the classifier
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
results = []
labels = []

start = time.time()
clickbait = pd.read_csv("resources/click_bait/clickbait.csv", usecols=['Video Title'])
clickbait['label'] = 1

not_clickbait = pd.read_csv("resources/click_bait/notClickbait.csv", usecols=['Video Title'])
not_clickbait['label'] = 0

concatenated = [clickbait, not_clickbait]

test_set = pd.concat(concatenated)

test_set.reset_index()

for index, line in test_set.iterrows():

    res = classifier(line['Video Title'])
    labels.append(line['label'])

    label = 1
    if "0" in res[0]["label"]:
        label = 0
    results.append(label)

print(results)
print(labels)
tn, fp, fn, tp = confusion_matrix(labels, results).ravel()

print("True Negative: ", tn)
print("False Negative: ", fn)
print("True Positive: ", tp)
print("False Positive: ", fp)
print()
print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
print("Elapsed time: ", time.time() - start)
