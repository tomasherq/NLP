import json

starters = []

with open("fake_news/fake_news.json", "r") as file_read:

    phrases = json.loads(file_read.read())

    for phrase in phrases:
        sentences = phrase.split(".")
        for sentence in sentences:
            words = (sentence.strip().split(" ")[:7])
            new_words = list()
            for word in words:
                new_word = ""
                for character in word:
                    if character.isalnum():
                        new_word += character
                if new_word != '':
                    new_words.append(new_word)

            if len(new_words) == 7:
                starters.append(new_words)


with open("starters.txt", "w") as file_write:
    for setOfWords in starters:
        file_write.write(" ".join(setOfWords)+"\n")
