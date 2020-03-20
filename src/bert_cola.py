from pathlib import Path
import torch
import pandas as pd
import random
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.cuda.random import manual_seed_all

RAW_FOLDER = Path('../glue_data/CoLA/original/raw')

MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 4


def flat_accuracy(preds, labs):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labs.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


device = torch.device('cuda:1')
df = pd.read_csv(RAW_FOLDER/'in_domain_train.tsv',
                 delimiter='\t', header=None,
                 names=['sentence_source', 'label', 'label_notes', 'sentence'])

sentences = df.sentence.values
labels = df.label.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)

# ----- Tokenization and Padding -----
input_ids = []

for sent in sentences:
    encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
    input_ids.append(encoded_sent)

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', 
                          value=0, truncating='post', padding='post')

# ----- Attention Masks -----
attention_masks = []

for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

# ----- Training and Validation Split -----
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, labels, random_state=2018, test_size=0.1)
train_masks, val_masks, _, _ = train_test_split(
    attention_masks, labels, random_state=2018, test_size=0.1)

# ----- Convert to Tensors -----
train_inputs = torch.tensor(train_inputs)
val_inputs = torch.tensor(val_inputs)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)

# ----- Create Datasets -----
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler,
                              batch_size=BATCH_SIZE)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler,
                            batch_size=BATCH_SIZE)

# ----- Model and Optimizer -----
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# ----- Training Loop -----
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
manual_seed_all(seed_val)

loss_values = []
model.zero_grad()

for epoch_i in range(0, EPOCHS):
    print('Epoch: {}'.format(epoch_i + 1))
    # ----- Training Step -----
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        model.train()

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    # ----- Validation Step -----
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
