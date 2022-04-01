import json
words = []
with open("wordlist.10000", "r") as file_read:

    for line in file_read.readlines():
        word = line.strip()
        if len(word) > 4:
            words.append(word)

with open("words.json", "w") as fw:

    fw.write(json.dumps(words, indent=4))
