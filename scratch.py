df.columns

text = tokenizer.tokenize(df.Text[0])
# os.pardir(root_path)

max_len = 512

text = text[:max_len-2]

text

input_sequence = ["[CLS]"] + text + ["[SEP]"]

pad_len = max_len-len(input_sequence)


tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len


# create an array of 1s that represent valid words, and append zeros that represent padded words
pad_masks = [1] * len(input_sequence) + [0] * pad_len

# create an array of zeros
segment_ids = [0] * max_len


all_tokens.append(tokens)
all_masks.append(pad_masks)
all_segments.append(segment_ids)

return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))
import transformers

import requests
import wget


