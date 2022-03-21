from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


class BaitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


fileLocation = "resources/click_bait_phrases.json"

sentences = []
labels = []
with open(fileLocation, "r") as file_read:

    dataset = json.load(file_read)

for data in dataset:
    sentences.append(data["text"])
    labels.append(data["label"])

training_size = int(len(sentences)*0.8)


train_texts = sentences[0:training_size]
train_labels = labels[0:training_size]

test_texts = sentences[training_size:]
test_labels = labels[training_size:]


train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

tokenizer = AutoTokenizer.from_pretrained('gpt2')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


train_dataset = BaitDataset(train_encodings, train_labels)
val_dataset = BaitDataset(val_encodings, val_labels)
test_dataset = BaitDataset(test_encodings, test_labels)


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)


model = AutoModelForSequenceClassification.from_pretrained("gpt2")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model("models")
