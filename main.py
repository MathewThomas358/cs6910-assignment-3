"""
    Assignment-3
    Reurrent Neural Networks (RNNs)
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim

import wandb as wb

from data import Data

# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO

print(f'Running on {DEVICE}')

DATA_DIR_PATH = r'data/mal/'
DATA_TRAIN_PATH = r'mal_train.csv'
DATA_TEST_PATH = r'mal_test.csv'
DATA_VALID_PATH = r'mal_valid.csv'

train_data = Data(DATA_DIR_PATH + DATA_TRAIN_PATH)
valid_data = Data(DATA_DIR_PATH + DATA_VALID_PATH)
test_data  = Data(DATA_DIR_PATH + DATA_TEST_PATH)

class EnteEncoder(nn.Module):

    def __init__(self, cell_type: str, input_size, embedding_size, hidden_size, num_layers, dropout, bidi):
        
        super(EnteEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.bidi = bidi

        assert cell_type is not None, "Provide a valid cell type"

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidi)
        
        elif cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidi)

        elif cell_type == "RNN":
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidi)

    def forward(self, x_trai):

        embedding = self.dropout(self.embedding(x_trai))

        if isinstance(self.rnn, nn.LSTM):
            _, (hidden, cell) = self.rnn(embedding)

            return hidden, cell

        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            _, hidden = self.rnn(embedding)
            # if self.bidi:
                # hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            return hidden

class EnteDecoder(nn.Module):

    def __init__(self, cell_type: str, input_size, embedding_size, hidden_size, output_size, num_layers, dropout, bidi):

        super(EnteDecoder, self).__init__()

        self.bidi = bidi
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # if self.num_layers == 1:
            # self.dropout = nn.Dropout(0)
        # else:
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(input_size, embedding_size)

        assert cell_type is not None, "Provide a valid cell type"

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidi)
        
        elif cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidi)

        elif cell_type == "RNN":
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidi)

        if self.bidi:
            self.fully_conn = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fully_conn = nn.Linear(hidden_size, output_size)

    def forward(self, x_trai, hidden, cell):

        x_trai = x_trai.unsqueeze(0)
        embedding = self.dropout(self.embedding(x_trai))

        if isinstance(self.rnn, nn.LSTM):
            output, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        elif isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            output, hidden = self.rnn(embedding, hidden)

        predictions = self.fully_conn(output)
        predictions = torch.softmax(predictions, dim = 2)
        predictions = predictions.squeeze(0)

        if isinstance(self.rnn, nn.LSTM):
            return predictions, hidden, cell
        elif isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            return predictions, hidden

class EnteSeq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device, bidi: bool):

        super(EnteSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.bidi = bidi

    def forward(self, source, target, tfr = 0.5):

        batch_siz = source.shape[1]
        target_len = target.shape[0]
        target_vocab_length = train_data.num_decoder_tokens #TODO total number of letters in malayalam

        outputs = torch.zeros(target_len, batch_siz, target_vocab_length).to(self.device)

        if isinstance(self.encoder.rnn, nn.GRU) or isinstance(self.encoder.rnn, nn.RNN):
            hidden = self.encoder(source)

        if isinstance(self.encoder.rnn, nn.LSTM):
            hidden, cell = self.encoder(source)

        x_targ = target[0]

        for count in range(1, target_len):

            if isinstance(self.decoder.rnn, nn.LSTM):
                out, hidden, cell = self.decoder(x_targ, hidden, cell)
            elif isinstance(self.decoder.rnn, nn.GRU) or isinstance(self.encoder.rnn, nn.RNN):
                out, hidden = self.decoder(x_targ, hidden, None)

            outputs[count] = out
            best = out.argmax(1)
            x_targ = target[count] if random.random() < tfr else best

        return outputs

class EnteTransliterator:

    def __init__(
        self,
        cell_type: str = "LSTM",
        epochs: int = 15,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        hidden_size: int = 128,
        lr: float = 1e-3,
        batch_size: int = 128,
        dropout: float = 0.5,
        bidirectional: bool = True,
        emb: int = 100
    ):

        self.cell_type = cell_type
        self.epochs = epochs

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        if encoder_layers != decoder_layers:
            self.encoder_layers = decoder_layers
        # if encoder_layers == 1 and decoder_layers != 1:
        #     self.encoder_layers = decoder_layers
        #     self.decoder_layers = decoder_layers
        # if encoder_layers != 1 and decoder_layers == 1:
        #     self.decoder_layers = encoder_layers
        #     self.encoder_layers = encoder_layers
        self.hidden_size = hidden_size
        self.learning_rate = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.embedding_size = emb


    def train(self):
        
        train_data.set_batch_size(self.batch_size)
        valid_data.set_batch_size(self.batch_size)
        input_size_encoder = train_data.num_encoder_tokens
        input_size_decoder = train_data.num_decoder_tokens

        total_batches = len(train_data.source) // self.batch_size

        encoder = EnteEncoder(
            self.cell_type,
            input_size_encoder,
            self.embedding_size,
            self.hidden_size,
            self.encoder_layers,
            self.dropout,
            self.bidirectional
        )

        decoder = EnteDecoder(
            self.cell_type,
            input_size_decoder,
            self.embedding_size,
            self.hidden_size,
            input_size_decoder,
            self.decoder_layers,
            self.dropout,
            self.bidirectional
        )

        model = EnteSeq2Seq(encoder, decoder, DEVICE, self.bidirectional).to(DEVICE)

        loss = nn.CrossEntropyLoss(ignore_index=train_data.target_chars_index[Data.pad])
        optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)

        accuracy = 0
        los = None

        val_accuracy = 0
        val_los = None

        for epoch in range(self.epochs):

            total_correct = 0
            total_samples = 0

            model.train()
            for batch in range(total_batches):

                input_data, target = train_data.get_batch(DEVICE)
                output = model(input_data, target).to(DEVICE)

                output = output[1:].reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)

                optimizer.zero_grad()
                los = loss(output, target)
                los.backward()
                optimizer.step()

                predicted = torch.argmax(output, dim=1)
                correct = (predicted == target).sum().item()
                total_correct += correct
                total_samples += target.size(0)

            accuracy = total_correct / total_samples
            model.eval()

            with torch.no_grad():
                val_total_correct = 0
                val_total_samples = 0

                for batch in range(total_batches):

                    val_input_data, val_target = valid_data.get_batch(DEVICE)
                    val_output = model(val_input_data, val_target).to(DEVICE)

                    val_output = val_output[1:].reshape(-1, val_output.shape[2])
                    val_target = val_target[1:].reshape(-1)

                    val_los = loss(val_output, val_target)

                    val_predicted = torch.argmax(val_output, dim=1)
                    val_correct = (val_predicted == val_target).sum().item()
                    val_total_correct += val_correct
                    val_total_samples += val_target.size(0)

            val_accuracy = val_total_correct / val_total_samples

            sample_in, sample_target = train_data.get_random_sample(0)
            sample_in = sample_in.to(DEVICE)
            sample_target = sample_target.to(DEVICE)
            pred = model(
                sample_in.unsqueeze(1),
                sample_target.unsqueeze(1)
            )

            print(
                "SampleIn", train_data.sequence_to_text(sample_in, True),
                "SampleTar", train_data.sequence_to_text(sample_target),
                "Pred", train_data.sequence_to_text(
                        torch.argmax(pred.squeeze(), dim=1)
                    )
            )

            print(f'Epoch: {epoch + 1} Accuracy: {accuracy * 100} ValAc: {val_accuracy * 100}')


        wb.log({"train_accuracy": 100 * accuracy})
        wb.log({"valid_accuracy": 100 * val_accuracy})
        wb.log({"train_loss": los})
        wb.log({"valid_loss": val_los})

    def test(self):
        pass

#! TODO : variable encoder decoder layers
#! TODO: attention
#! TODO: Inference models

print("Training from main")
ente = EnteTransliterator(
    cell_type="LSTM",
    bidirectional=True,
    batch_size=128,
    dropout=0.3,
    emb=150,
    epochs=25,
    hidden_size=1024,
    lr=1e-3,
    decoder_layers=2,
    encoder_layers=2
)

ente.train()
