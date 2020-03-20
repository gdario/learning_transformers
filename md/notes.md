Notes from the Transformer Library Documentation
================================================

Quickstart
----------

-   Only 3 abstractions: configuration, model and tokenizer.
-   If you want to extend the library, use PyTorch and inherit from the
    base classes.

### To further investigate (more advanced topics).

**TODO** understand the topics below:

-   "A simple/consistent way to add new tokens to the vocabulary and
    embeddings for fine-tuning"
-   "Simple ways to mask and prune transformer heads".

### Main concepts

-   Model classes are `torch.nn.Modules`.
-   Configuration classes. Usually use the pretrained ones, but you can
    modify the settings here.
-   Tokenizer: stores the vocabulary and allows *encoding/decoding*.

All classes can be installed from pre-trained instances and saved locally with `from_pretrained()`, `save_pretrained()`.

### Organization of the documentation

1.  **Main Classes**: this section details the three main classes plus some optimization related classes.
2.  **Package Reference**: this section details on the variants of each class for each model.

Note: the first example ("Who was Jim Henson?") actually uses *two*
sentences, and it uses the segments tensor to tell them apart. The
example uses a `BertModel` to show how to encode the inputs into the
hidden states, and a `BertForMaskedLM` to predict the masked token. The
`segments_tensors` containing the sentence indexing are used as a value
for the `token_type_ids` argument.

**TODO** understand what `token_type_ids` does. When shall we use `segments_tensors`?

Transformer models always output *tuples*. The first element of the
output of the `BertModel`, `outputs[0]` is the hidden state of the last
layer of the model, i.e., the encoded inputs. This is also a tuple of
shape `(batch_size, seq_length, hidden_dim)`. The hidden dimension can
be obtained as `model.config.hidden_size`.

### Going from indices to words

In the `BertForMaskedLM` example we have the following code:

```python
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
```

### OpenAI GPT-2

`GPT2LMHeadModel` generates the next token given an input.

### Using the past

GPT-2, XLNet and other models can use a `past` or `mems` attribute to
avoid re-computing key/value pairs when using sequential decoding. Look
at the example at the end of the
[quickstart](https://huggingface.co/transformers/quickstart.html).

## Model upload and sharing

It is now possible to upload and share models via the CLI that ships
with the library. You first need to create an account on [this
page](https://huggingface.co/join).

## Examples

-   Language Model fine-tuning: causal LM for GPT-2. Masked LM for BERT.
-   GLUE: examples contain distributed training and half-precision.
-   SQuAD: question answering with BERT/XLnet etc. Also with distributed
    training.
-   Multiple Choice.
-   NER: BERT on the CoNLL 2003 dataset (learn more).

### Language model fine tuning

### [TODO]{.todo .TODO} select a corpus of text (medical text? Something else?) and fine {#select-a-corpus-of-text-medical-text-something-else-and-fine}

tune BERT.

### GLUE (General Language Understanding Evaluation)

[GLUE website](https://gluebenchmark.com/). [GLUE
Tasks](https://gluebenchmark.com/tasks). Note that there is also a
[SuperGLUE](https://super.gluebenchmark.com/) benchmark, on more
difficult language understanding tasks.

### [TODO]{.todo .TODO} use the `run_glue.py` script after modifying the base class to {#use-the-run_glue.py-script-after-modifying-the-base-class-to}

cover multi-label classification.

## Loading Google AI or OpenAI pre-trained weights or PyTorch dump

Tokenizer classes contain the vocabulary and the encoding functions.
Model classes can be also created with `torch.save()`. Uncased
models/tokenizers strip accent markers and convert to lowercase before
WordPiece tokenization. When using an uncased model, make sure to pass
`--do_lower_case` to the training scripts. According to the
documentation:

> Typically, the Uncased model is better unless you know that case
> information is important for your task (e.g., Named Entity Recognition
> or Part-of-Speech tagging)

## Tokenizer class

This class, besides the expected functionality, has methods for adding
tokens to the vocabulary. Not clear if this is done automatically when
training. New tokens can be added with the `add_tokens` method.

### Methods

- `add_special_tokens`. Example `special_token_dict = {'cls_token': '<CLS>'}`.
- `add_tokens`. Example: `tokenizer.add_tokens(['new_tok1', 'new_tok2'])`.
- `batch_encode_plus`: **TODO** learn what's this.
- `decode`: converts a sequence of integer ids in a string. Can skip special tokens. `clean_up_tokenization_spaces` remove the spaces before punctuation marks introduced by the tokenizer. `True` by default.
- `encode`: converts a string in a sequence of ids using the tokenizer and the vocabulary. Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`. It allows for truncation, padding and can return either TF or PT tensors.
- `encode_plus`: returns a dictionary containing the encoded sequence plus additional information (the mask for sequence classification and the overflowing elements).

### Do we need Keras `pad_sequence`?

`tokenizer.encode` takes a string and
1. adds the special tokens.
2. truncates to the max length.
3. pads to max length.

We used Keras' `pad_sequence` 


Add a dictionary of special tokens.

### Configuration class

One can read a configuration from:

1. A dictionary.
2. A JSON file.
3. A pretrained model.

For example, one can load a `BertConfig` and print it as a JSON string as follows.

```python
from transformers import BertConfig
config = BertConfig()
config_string = config.to_json_string()
```

Now `config_string` contains all the model configuration settings as a JSON string.

## Community uploaded models

[This URL](https://huggingface.co/models) contains a list of community uploaded models, including BioBert v1.1 (at the time of writing, the most recent weights). To use it, simply write:

```python
tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
```

**TODO** Understand what `AutoTokenizer` and `AutoModel` are.

## Models

According to the documentation the model class has methods to

- Resize the input token embeddings when new tokens are added to the vocabulary.
- Prune the attention heads of the model.

**TODO** understand what this means.
