import time
import datetime
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lowercase=True)

def encode_sentence(sentence):
    return tokenizer.encode(sentence, add_special_tokens=True)


def pad_encoded_sentence(encoded, maxlen):
    encoded.extend([0] * (maxlen - len(encoded)))
    return encoded


def process_sentence(sentence, maxlen):
    encoded = encode_sentence(sentence)
    padded = pad_encoded_sentence(encoded, maxlen)
    return padded


def process_sentences(sentences, maxlen):
    out = [process_sentence(sentence, maxlen) for sentence in sentences]
    return np.array(out, dtype='int64')


def create_attention_mask(inputs):
    out = np.where(inputs > 0, 0, 1)
    return out.astype('int64')


def flat_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
