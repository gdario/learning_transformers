from transformers import BertTokenizer

text = 'This is one sentence. Two actually'
text_batch = ['This is a sentence.',
              'This is another one',
              'What about some more?']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(text)

encoded_batch = tokenizer.batch_encode_plus(text_batch,
                                            add_special_tokens=True,
                                            max_length=24,
                                            return_attention_masks=True)
print(encoded_batch)
