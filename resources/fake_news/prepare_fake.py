import json
import csv
from textwrap import indent
import os
texts = []
DIRECTORY_NEWS = "total"

# Read from one dataset
for filename in os.listdir(DIRECTORY_NEWS):

    with open(f'{DIRECTORY_NEWS}/{filename}', 'r') as file_read:
        texts.append(file_read.read())


with open("fake_news.json", "w") as file_write:
    file_write.write(json.dumps(texts, indent=4))
