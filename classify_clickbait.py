from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import json

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("models/predict")
MAX_LENGTH = 20

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

with open("classified_phrases.json", "r") as file_read:

    classifiedPhrases = json.load(file_read)


os.listdir(f"output_phrases/{MAX_LENGTH}")


result = classifier("Meet the happiest #dog in the world!")
result2 = classifier("Tokyo's subway is shut down amid fears over an imminent North Korean missile attack on Japan")


print(result)
print(result2)
