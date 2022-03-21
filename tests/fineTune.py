import json
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from transformers import Trainer
import tensorflow as tf
fileLocation = "resources/click_bait_phrases.json"
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


sentences = []
labels = []
with open(fileLocation, "r") as file_read:

    dataset = json.load(file_read)

for data in dataset:
    sentences.append(data["text"])
    labels.append(data["label"])

training_size = int(len(sentences)*0.8)


training_sentences = sentences[0:training_size]
validation_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
validation_labels = labels[training_size:]


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_train = tokenizer(training_sentences, truncation=True, padding=True)
tokenized_validation = tokenizer(validation_sentences, truncation=True, padding=True)


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(tokenized_train),
    training_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(tokenized_validation),
    validation_labels
))

print(train_dataset)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics

)

trainer.train()
trainer.save_model("models")
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
