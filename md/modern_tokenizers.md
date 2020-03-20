# Modern Tokenizers

There is an excellent introduction to tokenization algorithms in [this blog post](https://blog.floydhub.com/tokenization-nlp/).

## Subword tokenization

In this approach, common words are tokenized as whole words, while less common words will be broken into smaller chunks. The size of the vocabulary is an important consideration.

- Larger vocabulary: more common words.
- Smaller vocabulary: more subword tokens.

## Byte Pair Encoding (BPE)

There is quite a bit of information in [this blog post](https://leimao.github.io/blog/Byte-Pair-Encoding/).

The main goal of the BPE subword algoritm is to find a way to represent a text with the least amount of tokens. We initially count the number of tokens, and then merge *byte pairs* based on their frequency. For example, if a character corresponds exactly to a byte (true only for ASCII), we:

1. Sort the characters by frequency.
2. Merge the two most frequent characters into a 2-gram.
3. Recompute the frequency of the characters, *subtracting* the frequency of the new 2-gram from the frequency of the two composing characters.
4. Repeat.

By iterating this process we initially increase the number of tokens, but this decreases as we aggregate more. Given enough iterations, this process will re-create the original vocabulary. The idea is to stop when we reach a pre-defined number of tokens (**TODO** verify that this is correct).

## Probabilistic Subword Tokenization

The above approach is a *greedy* one, and it can lead to ambiguities, as the same word can sometimes be obtained from different combinations of different subtokens. The differnt combinations would be associated with different embeddings, so we would have multiple embeddings for the same word.

### Unigram Subword Tokenization

This approach mimics a Language Model, where we try to predict the probability of the current word. The approach is described in [this paper](https://arxiv.org/pdf/1804.10959.pdf).

## WordPiece

This is the tokenizer used in BERT. It is similar to BPE, but rather than considering the frequence, it considers the change in likelihood after merging the tokens. This is the probability of the new merged pair minus the probability of both individual tokens occurring individually. It is still a greedy approach, but unlike BPE, it is based on probability, not frequency.

## SentencePiece

Described in [this paper](https://arxiv.org/abs/1808.06226).
