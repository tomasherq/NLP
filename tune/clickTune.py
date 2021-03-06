import re
import json

from datasets import load_metric
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead

# Define the paths where the datasets will be stored for the models
train_path = 'train_dataset_click.txt'
test_path = 'test_dataset_click.txt'

# Load the clickbait phrases
with open('../resources/click_bait/click_bait_phrases.json', "r") as f:
    data = json.load(f)

# Tokenizer to be used
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def load_dataset(train_path, test_path, tokenizer):
    # This is to load the datasets to be used!
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    # The data collator creates batches withthe data that is fed to it
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


# Function to save the files with the format we need
def build_text_files(data_json, dest_path):
    f = open(dest_path, 'w', encoding='utf-8')
    data = ''
    for texts in data_json:

        if texts["label"] == True:
            summary = str(texts['text']).strip()
            summary = re.sub(r"\s", " ", summary)
            data += summary + "  "
    f.write(data)

# Metrics we want to measure for the model (not used in this case)


def compute_metrics(eval_pred):

    metric = load_metric("accuracy")

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


train, test = train_test_split(data, test_size=0.3)
build_text_files(train, train_path)
build_text_files(test, test_path)


# We get the train and test datasets
train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

# Load the model
model = AutoModelWithLMHead.from_pretrained("gpt2")

# Set the trainning arguments
training_args = TrainingArguments(
    output_dir="../models/click_bait",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=32,  # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps=400,  # Number of update steps between two evaluations.
    save_steps=800,  # after # steps model is saved
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
)

# Start the trainer, then train the model and save it.
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
