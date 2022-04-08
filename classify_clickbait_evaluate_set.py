from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import csv
from sklearn.metrics import confusion_matrix
import time

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("models/predict")

# Create the classifier
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
results = []
labels = []

start = time.time()
with open('resources/click_bait/evaluate_clickbait.csv', "r", encoding="utf8") as f:
    lines = csv.reader(f)
    next(lines, None)
    for line in lines:
        res = classifier(line[0])
        labels.append(int(line[1]))

        label = 1
        if "0" in res[0]["label"]:
            label = 0
        results.append(label)

    tn, fp, fn, tp = confusion_matrix(labels, results).ravel()

    print("True Negative: ", tn)
    print("False Negative: ", fn)
    print("True Positive: ", tp)
    print("False Positive: ", fp)
    print()
    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    print("Elapsed time: ", time.time() - start)
