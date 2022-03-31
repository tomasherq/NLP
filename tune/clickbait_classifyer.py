import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from transformers import create_optimizer
import tensorflow as tf


def preprocess(examples):
    return tokenizer(examples["headline"], truncation=True)


def preprocess2(examples):
    return tokenizer(examples["text"], truncation=True)


if __name__ == '__main__':
    # df = pd.read_csv("../resources/clickbait_data.csv")
    imdb = load_dataset("imdb")
    # X = df["headline"]
    # y = df["clickbait"]

    # x_train, x_test = train_test_split(df, test_size=0.3, shuffle=True)

    # dataset = Dataset.from_pandas(df)

    # dataset = dataset.train_test_split(test_size=0.3)

    # dataset = load_dataset('csv', data_files="../resources/clickbait_data.csv")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    # tokenized_clickbait = dataset.map(preprocess, batched=True)
    tokenized_imdb = imdb.map(preprocess2, batched=True)

    tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_dataset = tokenized_imdb["test"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    model.compile(optimizer=optimizer)

    model.fit(x=tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)

