"""
    Assignment-3
    Reurrent Neural Networks (RNNs)
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data import Data

# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO

DATA_DIR_PATH = r'data/mal/'
DATA_TRAIN_PATH = r'mal_train.csv'
DATA_TEST_PATH = r'mal_test.csv'
DATA_VALID_PATH = r'mal_valid.csv'

train_data = Data(DATA_DIR_PATH + DATA_TRAIN_PATH)

class EnteEncoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        
        super(EnteEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x_trai):
        #x will be a vector of indices

        embedding = self.dropout(self.embedding(x_trai))
        _, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class EnteDecoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):

        super(EnteDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout)
        self.fully_conn = nn.Linear(hidden_size, output_size)

    def forward(self, x_trai, hidden, cell):

        x_trai = x_trai.unsqueeze(0)
        embedding = self.dropout(self.embedding(x_trai))

        out, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        predictions = self.fully_conn(out)
        predictions = torch.softmax(predictions, dim=2)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class EnteS2S(nn.Module):

    def __init__(self, encoder, decoder, device):

        super(EnteS2S, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, tfr = 0.5):

        batch_siz = source.shape[1]
        target_len = target.shape[0]
        target_vocab_length = train_data.num_decoder_tokens #TODO total number of letters in malayalam

        outputs = torch.zeros(target_len, batch_siz, target_vocab_length).to(self.device)

        hidden, cell = self.encoder(source)

        x_targ = target[0]

        for count in range(1, target_len):
            out, hidden, cell = self.decoder(x_targ, hidden, cell)
            outputs[count] = out
            best = out.argmax(2).squeeze()
            x_targ = target[count] if random.random() < tfr else best

        return outputs


epochs = 10
learning_rate = 0.001
batch_size = 128

train_data.set_batch_size(batch_size)
input_size_encoder = train_data.num_encoder_tokens
input_size_decoder = train_data.num_decoder_tokens

embedding_size = 100 #TODO configurable

# total_batches = 1 # TODO
total_batches = len(train_data.source) // batch_size # TODO

hidden_size = 1024 # sweep param
num_layers = 2 # sweep param

dropout = 0.5

encoder = EnteEncoder(
            input_size_encoder,
            embedding_size,
            hidden_size,
            num_layers,
            dropout
        )

decoder = EnteDecoder(
            input_size_decoder,
            embedding_size,
            hidden_size,
            input_size_decoder,
            num_layers,
            dropout
        )

model = EnteS2S(encoder, decoder, DEVICE).to(DEVICE)

loss = nn.CrossEntropyLoss(ignore_index=train_data.target_chars_index[Data.pad])
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):

    for batch in range(total_batches):

        input_data, target = train_data.get_batch(DEVICE)
        output = model(input_data, target).to(DEVICE)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        los = loss(output, target)
        los.backward()

    print(f'Epoch: {epoch + 1} Loss: {los}')
