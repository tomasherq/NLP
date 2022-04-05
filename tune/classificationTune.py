
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, DatasetDict, load_metric
import numpy as np

# The test dataset is 25% of the phrases
# We load the dataset and split it in both trainning and test for tunning the model
dataset = load_dataset('csv', data_files="../resources/click_bait/clickbait_data.csv", split='train')
train_testvalid = dataset.train_test_split()
test_valid = train_testvalid['test'].train_test_split()
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],  # 0.75
    'test': test_valid['test']})  # 0.25

# Create the tokenizer for the model we want to train
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Test the accuracy of the trained model
def compute_metrics(eval_pred):

    metric = load_metric("accuracy")

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


# Obtain the tokenized dataset and split for test and train
tokenized_datasets = train_test_valid_dataset.map(tokenize_function, batched=True)

# Get the datasets for trainning and evaluation
small_train_dataset = tokenized_datasets["train"].shuffle(seed=28).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=28).select(range(1000))

# Get the model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Establish the trainning arguments, the directory to store the model and the evaluation
training_args = TrainingArguments(output_dir="../models/predict", evaluation_strategy="epoch")

# Create the trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model()
