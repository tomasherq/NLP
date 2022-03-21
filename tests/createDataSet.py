from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import json


with open("resources/click_bait_phrases.json", 'r') as f:
    datastore = json.load(f)

# First we load the phrases
sentences = []
labels = []
for item in datastore:
    sentences.append(item['text'])
    labels.append(item['label'])

training_size = int(len(sentences)*0.8)


# Divide them between trainning and validation
training_sentences = sentences[0:training_size]
validation_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
validation_labels = labels[training_size:]

# Tokenize the phrases
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(training_sentences, truncation=True, padding=True)
val_encodings = tokenizer(validation_sentences, truncation=True, padding=True)

# Create the tensorflow datasets we are going to feed
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), training_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), validation_labels))

# We classify two labels in this example. In case of multiclass
# classification, adjust num_labels value
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Now we optimize the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])


model.fit(train_dataset, validation_data=val_dataset, epochs=3)
model.save_pretrained("models/click_bait_custom_model")
