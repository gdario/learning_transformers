import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging

logging.basicConfig(level=logging.INFO)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token for BertForMaskedLM ('puppeteer')
masked_index = 8

tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]',
                          'jim', '[MASK]', 'was', 'a', 'puppet', '##eer',
                          '[SEP]']

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Think of how you can automate this.
segments_ids = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensor = torch.tensor([segments_ids])

# Run BertModel
model = BertModel.from_pretrained('bert-base-uncased')

model.eval()

tokens_tensor = tokens_tensor.to('cuda')
segments_tensor = segments_tensor.to('cuda')
model.to('cuda')

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensor)
    encoded_layers = outputs[0]

print(encoded_layers)
model.to('cpu')

# Run BertForMaskedLM

model_lm = BertForMaskedLM.from_pretrained('bert-base-uncased')
model_lm.eval()
model_lm.to('cuda')

with torch.no_grad():
    outputs_lm = model_lm(tokens_tensor, token_type_ids=segments_tensor)
    predictions = outputs_lm[0]

predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
