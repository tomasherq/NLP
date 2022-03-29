import tensorflow as tf
from transformers import pipeline
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import sys
import json
import os

if len(sys.argv) < 2:
    print("Introduce the length of the phrases.")
    exit()

MAX_LENGTH = (sys.argv[1])

directoryOutput = f'output_phrases/{MAX_LENGTH}'

if not os.path.exists(directoryOutput):
    os.makedirs(directoryOutput)

filename = len(os.listdir(directoryOutput))+1

if(filename > 1000):
    exit()


with open("resources/starters/used_phrases.json", "r") as file_read:
    used_phrases = json.load(file_read)

if MAX_LENGTH not in used_phrases.keys():
    used_phrases[MAX_LENGTH] = []

initialText = ''
with open("resources/starters/starters.txt", "r") as file_read:
    for line in file_read.readlines():
        if line not in used_phrases[MAX_LENGTH]:
            used_phrases[MAX_LENGTH].append(line)
            initialText = line
            break

# Create a news article from a clickbait title?


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("./models/old_bait", pad_token_id=tokenizer.eos_token_id, from_pt=True)


# encode context the generation is conditioned on
input_ids = tokenizer.encode(initialText, return_tensors='tf')

# This way I get the same seed for all lengths and filenames ;)
tf.random.set_seed(int(MAX_LENGTH)+filename)


# generate text until the output length (which includes the context length) reaches 50
# beam_output = model.generate(
#     input_ids,
#     max_length=MAX_LENGTH,
#     num_beams=5,
#     no_repeat_ngram_size=2,
#     num_return_sequences=5,
#     early_stopping=True
# )

initialText = initialText

# # Top k
# sample_outputs = model.generate(
#     input_ids,
#     do_sample=True,
#     max_length=MAX_LENGTH,
#     top_k=50
# )


# Top p
sample_outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=int(MAX_LENGTH),
    top_k=50,
    top_p=0.95,
    num_return_sequences=5
)


output_texts = []

with open(f"{directoryOutput}/{filename}.json", "w") as file_write:
    for beam_output in (sample_outputs):
        output_texts.append(tokenizer.decode(beam_output, skip_special_tokens=False).split("<|endoftext|>")[0])
    file_write.write(json.dumps(output_texts, indent=4))


with open("resources/starters/used_phrases.json", "w") as file_write:
    file_write.write(json.dumps(used_phrases, indent=4))
