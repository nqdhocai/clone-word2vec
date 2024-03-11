import math

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class Tokenizer:
    def __init__(self, max_length = 50, vocab_size = 1000):
        self.vocab = {}
        self.vocab_size = vocab_size
        self.oov_token = None
        self.max_length = max_length

    def fit(self, sentences, oov_token='<OOV>'):
        self.oov_token = oov_token
        stop_word = ['.', '!', '?']
        tokens = []
        for sentence in sentences:
            for i in stop_word:
                sentence.replace(i, '')
            tokens.extend(sentence.split(' '))

        words = list(set(tokens))
        vocab = {word: 0 for word in words}
        vocab[oov_token] = 0

        for token in tokens:
            if token not in words:
                vocab[oov_token] += 1
            else:
                vocab[token] += 1

        vocab = list(sorted(vocab.items(), key=lambda item: item[1]))
        count = 0
        for rank in range(1, len(vocab) + 1):
            if count == self.vocab_size:
                break
            word = vocab[rank - 1][0]
            self.vocab[word] = rank
            count += 1

    def transform(self, sentences):
        tokened = []
        for sentence in sentences:
            token = sentence.split(' ')
            for i in range(len(token)):
                word = token[i]
                if word not in self.vocab:
                    token[i] = self.vocab[self.oov_token]
                else:
                    token[i] = self.vocab[word]
            tokened.append(token)
        return self.add_padding(tokened)

    def add_padding(self, tokened):
        max_length = self.max_length
        tokened_size = len(tokened)
        for i in range(tokened_size):
            if len(tokened[i]) < max_length:
                pad = [0 for _ in range(max_length - len(tokened[i]))]
                pad.extend(tokened[i])
                tokened[i] = pad

            else:
                tokened[i] = tokened[i][:max_length]
        return tokened

class Embedding(nn.Module):
    def __init__(self, embedding_dim, max_length = 50, vocab_size = 1000):
        super(Embedding, self).__init__()
        self.tokenizer = Tokenizer(max_length, vocab_size)
        input_size = self.tokenizer.max_length
        self.fc1 = nn.Linear(input_size, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, input_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

    def dataset(self, sentences):
        self.tokenizer.fit(sentences)
        data = self.tokenizer.transform(sentences)
        for pos, vec in enumerate(data):
            for i, num in enumerate(vec):
                temp = [0 for _ in range(self.tokenizer.vocab_size)]
                if num == 0:
                    vec[i] = temp
                x = num
                vec[i] = temp.copy()
                vec[i][x-1] = 1
        x_train = torch.tensor(data, dtype=torch.float).permute(0, 2, 1)
        for pos, vec in enumerate(data):
            vec.pop(-1)
            vec.insert(0, [0 for _ in range(self.tokenizer.vocab_size)])
            data[pos] = vec

        y_train = torch.tensor(data, dtype=torch.float).permute(0, 2, 1)

        return x_train, y_train

    def fit(self, sentences, num_epochs=10, batch_size=64, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        x_train, y_train = self.dataset(sentences)

        current_loss=100
        for epoch in range(num_epochs):
            for i in range(len(x_train)):
                inputs = x_train[i]
                targets = y_train[i]

                # Lan truyền tiến
                outputs = self(inputs)

                # Tính toán độ lỗi
                loss = criterion(outputs, targets)

                # Lan truyền ngược và cập nhật trọng số
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if math.fabs(loss.item() - current_loss) <= 10 ** -8:
                break
                # Update Current loss
            current_loss = loss.item()

    def transform(self, x):
        out = torch.tensor(self.tokenizer.transform(x), dtype=torch.float)
        out = self.fc1(out)
        return out


# Load data
data = pd.read_json('E:\CODE-Codespace\JB-Pycharm\\train_data.json')
data = data.drop(columns=['id', 'diagramRef'])

sentences = [
    'toi ten la dung.',
    'toi hoc dai hoc bkhn!'
]
vocab_size = 100

test = ['mai la thu 3']
embedd = Embedding(5)
embedd.fit(sentences)
print(embedd.transform(test))