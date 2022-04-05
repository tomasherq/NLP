from transformers import TFGPT2LMHeadModel, AutoTokenizer
import json
import os
from random import randint

# Variable to set the maximum length of the sentences.
MAX_LENGTH = 10
FILE_LIMIT = 1200

# Where to store the resulting clickbait phrases.
directoryOutput = f'output_phrases/{MAX_LENGTH}'

if not os.path.exists(directoryOutput):
    os.makedirs(directoryOutput)

# Keep a record of the words that have been already used.
with open("resources/starters/used_phrases.json", "r") as file_read:
    used_phrases = json.load(file_read)

if MAX_LENGTH not in used_phrases.keys():
    used_phrases[MAX_LENGTH] = []


# Load the tokenizer and the model that we tuned.
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = TFGPT2LMHeadModel.from_pretrained("./models/click_bait", pad_token_id=tokenizer.eos_token_id, from_pt=True)

# We use 1200 words to create 5 phrases with each.
i = 0
while i < FILE_LIMIT:

    filename = len(os.listdir(directoryOutput))+1

    if filename > FILE_LIMIT:
        exit(1)

    # We search for the word that we want.
    initialText = ''
    with open("resources/starters/words.json", "r") as file_read:

        # Search for the word to be used to create the phrases.
        words = json.load(file_read)
        randomNumber = randint(0, len(words)-1)
        while(True):
            # We set the inital text nd is only valid if it
            # has been used before.

            initialText = words[randomNumber]
            if initialText not in used_phrases[MAX_LENGTH]:
                used_phrases[MAX_LENGTH].append(initialText)
                break
            randomNumber += 1

    # We encode the input ids for the text.
    input_ids = tokenizer.encode(initialText, return_tensors='tf')

    # The top_k=0 makes the options for the next token incredibly wide,
    # as we do not want to create a long coherent text but a title
    # it only makes sense to set it 0.
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,             # We do sampling, so we will not use greeding decoding.
        max_length=int(MAX_LENGTH),
        top_k=0,                    # Most important parameter so we have all vocabulary tokens available for text creation*
        num_return_sequences=5,     # We return 5 sentences.
        # Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
        early_stopping=True
    )

    # Store the generated texts and save them to a .json file.
    output_texts = []

    with open(f"{directoryOutput}/{filename}.json", "w") as file_write:
        for beam_output in (sample_outputs):
            text_clean = tokenizer.decode(beam_output, skip_special_tokens=True).strip().capitalize()

            output_texts.append(text_clean)
        file_write.write(json.dumps(output_texts, indent=4))

    # Store the word that has been used so we do not use it again.
    with open("resources/starters/used_phrases.json", "w") as file_write:
        file_write.write(json.dumps(used_phrases, indent=4))
    i += 1
