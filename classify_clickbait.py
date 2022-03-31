from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("models/predict")

classifier = pipeline("text-classification", model= model, tokenizer= tokenizer)

result = classifier("Meet the happiest #dog in the world!")
result2 = classifier("Tokyo's subway is shut down amid fears over an imminent North Korean missile attack on Japan")


print(result)
print(result2)