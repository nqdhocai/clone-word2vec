import collections
import torch
import torch.nn as nn
import torch.optim as optim


class Vocab:
    def __init__(self, corpus, min_freq=0, token='word'):
        tokens = self.tokenize(corpus, token=token)
        reserved_tokens = []
        # Sort according to frequencies
        counter = self.count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def count_corpus(self, sentences):
        # Flatten a list of token lists into a list of tokens
        tokens = [tk for line in sentences for tk in line]
        return collections.Counter(tokens)

    def tokenize(self, corpus, token='word'):
        """Split sentences into word or char tokens."""
        if token == 'word':
            return [sentence.split(' ') for sentence in corpus]
        elif token == 'char':
            return [list(sentence) for sentence in corpus]
        else:
            print('ERROR: unknown token type ' + token)


class Tokenizer:
    def __init__(self, vocab=None, token='word', max_length=32):
        self.token = token
        self.max_length = max_length
        if vocab != None:
            self.vocab = vocab
        else:
            self.vocab = Vocab(corpus=[], token=token)

    def tokenize(self, corpus, token='word'):
        """Split sentences into word or char tokens."""
        if token == 'word':
            return [sentence.split(' ') for sentence in corpus]
        elif token == 'char':
            return [list(sentence) for sentence in corpus]
        else:
            print('ERROR: unknown token type ' + token)

    def transform(self, sentences):
        tokens = self.tokenize(sentences, token=self.token)
        encode_tokens = []
        for token in tokens:
            temp = self.vocab[token]
            if len(temp) >= self.max_length:
                temp = temp[:self.max_length]
                encode_tokens.append(temp)
                continue

            else:
                temp.extend([0]*(self.max_length-len(temp)))
                encode_tokens.append(temp)
        return encode_tokens
