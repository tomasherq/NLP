import tensorflow as tf
from transformers import pipeline
from transformers import TFGPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead
import sys
import json
import os
import requests
from random import randint

MAX_LENGTH = 50
directoryOutput = f'output_phrases/{MAX_LENGTH}'

if not os.path.exists(directoryOutput):
    os.makedirs(directoryOutput)


with open("resources/starters/used_phrases.json", "r") as file_read:
    used_phrases = json.load(file_read)

if MAX_LENGTH not in used_phrases.keys():
    used_phrases[MAX_LENGTH] = []

i = 0

# Create a news article from a clickbait title?

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = TFGPT2LMHeadModel.from_pretrained("./models/click_bait", pad_token_id=tokenizer.eos_token_id, from_pt=True)

while i < 1000:

    filename = len(os.listdir(directoryOutput))+1

    if filename > 1000:
        exit(1)

    initialText = ''
    with open("resources/starters/words.json", "r") as file_read:

        words = json.load(file_read)
        randomNumber = randint(0, len(words)-1)
        while(True):

            initialText = words[randomNumber]
            if words[randomNumber] not in used_phrases[MAX_LENGTH]:
                initialText = words[randomNumber]

                break
            randomNumber += 1
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(initialText, return_tensors='tf')
    # Top p
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=int(MAX_LENGTH),
        top_k=0,
        num_return_sequences=5,
        early_stopping=True
    )

    output_texts = []

    with open(f"{directoryOutput}/{filename}.json", "w") as file_write:
        for beam_output in (sample_outputs):
            text_clean = tokenizer.decode(beam_output, skip_special_tokens=True).strip().capitalize()

            output_texts.append(text_clean)
        file_write.write(json.dumps(output_texts, indent=4))

    with open("resources/starters/used_phrases.json", "w") as file_write:
        file_write.write(json.dumps(used_phrases, indent=4))
    i += 1
