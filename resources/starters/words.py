import json


# Open the word list and only save words that are more than
# 4 characters long
words = []
with open("wordlist.10000", "r") as file_read:

    for line in file_read.readlines():
        word = line.strip()
        if len(word) > 4:
            words.append(word)

# Store the selected words
with open("words.json", "w") as fw:

    fw.write(json.dumps(words, indent=4))
