import tensorflow as tf
from transformers import pipeline
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import sys


# Create a news article from a clickbait title?

initialText = 'Pool party gone wrong.[My gf goes mad]'

if len(sys.argv) < 2:
    MAX_LENGTH = sys.argv[1]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("./models/click_bait", pad_token_id=tokenizer.eos_token_id, from_pt=True)


# encode context the generation is conditioned on
input_ids = tokenizer.encode(initialText, return_tensors='tf')

tf.random.set_seed(0)


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
    max_length=MAX_LENGTH,
    top_k=50,
    top_p=0.95,
    num_return_sequences=5
)


print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=False)))
