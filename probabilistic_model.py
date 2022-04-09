# The code used in this probabilistic model is based on:
# https://www.kaggle.com/code/amananandrai/clickbait-detector-naive-bayes-classifier/data
import json
import os
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string as s
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

nltk.download('wordnet')
nltk.download('omw-1.4')


def load_json_file(filename):
    with open(filename, "r") as file_read:
        return json.load(file_read)


def write_json(filename, dict_to_print):
    with open(filename, "w") as file_write:
        return file_write.write(json.dumps(dict_to_print, indent=4))


start_time = time.time()

# Read the training and test data to use
train_data = pd.read_csv('resources/click_bait/clickbait_data.csv')
test_data = pd.read_csv('resources/click_bait/evaluate_clickbait.csv')

# Making two Series with the clickbait and label for the training data
train_clickbait = train_data.text
train_labels = train_data.label

# Making two Series with the clickbait and label for the testing data
test_clickbait = test_data.iloc[:, 0]
test_labels = test_data.iloc[:, 1]


# Splits the click-bait string into a list of tokens
def tokenization(text):
    lst = text.split()
    return lst


# Makes all words lower case in the click-bait
def lowercasing(lst):
    new_lst = []
    for i in lst:
        i = i.lower()
        new_lst.append(i)
    return new_lst


# Removes punctuation in the click-bait
def remove_punctuations(lst):
    new_lst = []
    for i in lst:
        for j in s.punctuation:
            i = i.replace(j, '')
        new_lst.append(i)
    return new_lst


# Removes numbers in the click-bait
def remove_numbers(lst):
    nodig_lst = []
    new_lst = []
    for i in lst:
        for j in s.digits:
            i = i.replace(j, '')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i != '':
            new_lst.append(i)
    return new_lst


# Removes stopwords in the English language in the click-bait
def remove_stopwords(lst):
    stop = stopwords.words('english')
    new_lst = []
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst


# Removes extra spaces in the click-bait
def remove_spaces(lst):
    new_lst = []
    for i in lst:
        i = i.strip()
        new_lst.append(i)
    return new_lst


# Transform all words to their root form such that they are analysed as a single item
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatzation(lst):
    new_lst = []
    for i in lst:
        i = lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst


# Applying data transformations to training data
train_clickbait = train_clickbait.apply(tokenization) \
    .apply(lowercasing) \
    .apply(remove_punctuations) \
    .apply(remove_numbers) \
    .apply(remove_stopwords) \
    .apply(remove_spaces) \
    .apply(lemmatzation)

# Applying data transformations to test data
test_clickbait = test_clickbait.apply(tokenization) \
    .apply(lowercasing) \
    .apply(remove_punctuations) \
    .apply(remove_numbers) \
    .apply(remove_stopwords) \
    .apply(remove_spaces) \
    .apply(lemmatzation)

# Turning tokens list to string again
train_clickbait = train_clickbait.apply(lambda x: ''.join(i + ' ' for i in x))
test_clickbait = test_clickbait.apply(lambda x: ''.join(i + ' ' for i in x))

# Using TfidfVectorizer to convert click-bait text into features
tfidf = TfidfVectorizer()
train_clickbait_1 = tfidf.fit_transform(train_clickbait)
test_clickbait_1 = tfidf.transform(test_clickbait)

print(f"Number of features extracted: {len(tfidf.get_feature_names())}")
# print("The 100 features extracted from TF-IDF ")
# print(tfidf.get_feature_names()[:100])

# print("Shape of train set", train_clickbait_1.shape)
# print("Shape of test set", test_clickbait_1.shape)

# Preparing data for Naive Bayes Classifier
train_arr = train_clickbait_1.toarray()
test_arr = test_clickbait_1.toarray()

# Training the model
NB_MN = MultinomialNB()
NB_MN.fit(train_arr, train_labels)

# Performing prediction
print(test_arr)
pred = NB_MN.predict(test_arr)

end_time = time.time()

# print('First actual labels (limit 20 labels):\t\t', test_labels.tolist()[:100])
# print('First predicted labels (limit 20 labels):\t', pred.tolist()[:100])

print(f"Accuracy of the model is {accuracy_score(test_labels, pred)}")
print(f"Finished in {end_time - start_time} seconds")

MAX_LENGTH = 15

# File to keep record of the classified files
classifiedFile = f"classification/memory/classified_phrases_prob_{MAX_LENGTH}.json"
classifiedPhrases = []
if os.path.exists(classifiedFile):
    classifiedPhrases = load_json_file(classifiedFile)

# File to keep record of all the labels for the phrases
labelsFile = f"classification/results/labels_prob_{MAX_LENGTH}.json"
labels = []
if os.path.exists(labelsFile):
    labels = load_json_file(labelsFile)


# Directory of the phrases
phrasesDirectory = f"output_phrases/{MAX_LENGTH}"
for file in os.listdir(phrasesDirectory):

    if file not in classifiedPhrases:
        file_location = phrasesDirectory+"/"+file
        sentences = load_json_file(file_location)

        # Analyze if the phrase is clickbait
        for sentence in sentences:
            print(sentence)
            result = NB_MN.predict(sentence)
            print(result)
            label = 1
            if "0" in result["label"]:
                label = 0

            labels.append(label)

        # Write the results and that the file has been analyzed
        classifiedPhrases.append(file)

        write_json(classifiedFile, classifiedPhrases)
        write_json(labelsFile, labels)
