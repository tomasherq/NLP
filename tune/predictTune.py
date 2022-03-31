
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, DatasetDict, load_metric
import numpy as np

dataset = load_dataset('csv', data_files="../resources/click_bait/clickbait_data.csv", split='train')
train_testvalid = dataset.train_test_split()
test_valid = train_testvalid['test'].train_test_split()
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test']})


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):

    metric = load_metric("accuracy")

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


tokenized_datasets = train_test_valid_dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
training_args = TrainingArguments(output_dir="../models/predict", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model()
