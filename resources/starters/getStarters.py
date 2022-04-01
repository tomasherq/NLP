import json

starters = []

with open("../click_bait/click_bait_phrases.json", "r") as file_read:

    phrases = json.loads(file_read.read())

    for phrase in phrases:
        if phrase['label'] == False:
            sentences = phrase['text'].split(".")
            for sentence in sentences:
                words = (sentence.strip().split(" ")[:5])
                new_words = list()
                for word in words:
                    new_word = ""
                    for character in word:
                        if character.isalnum():
                            new_word += character
                    if new_word != '':
                        new_words.append(new_word)
                new_words.sort()
                if len(new_words) == 5:
                    if new_words not in starters:

                        starters.append(new_words)


with open("starters.txt", "w") as file_write:
    for setOfWords in starters:
        file_write.write(" ".join(setOfWords)+"\n")
