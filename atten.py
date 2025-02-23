"""

"""

import random
import datetime as time

import torch
import torch.nn as nn
import torch.optim as optim

import wandb as wb

from data import Data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR_PATH = r'data/mal/'
DATA_TRAIN_PATH = r'mal_train.csv'
DATA_TEST_PATH = r'mal_test.csv'
DATA_VALID_PATH = r'mal_valid.csv'

train_data = Data(DATA_DIR_PATH + DATA_TRAIN_PATH)
valid_data = Data(DATA_DIR_PATH + DATA_VALID_PATH)
test_data  = Data(DATA_DIR_PATH + DATA_TEST_PATH)

class EnteEncoder(nn.Module):

    def __init__(self, cell_type: str, input_size, embedding_size, hidden_size, dropout):
        
        super(EnteEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.bidi = True

        assert cell_type is not None, "Provide a valid cell type"

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, 1, dropout=dropout, bidirectional=self.bidi)
        
        elif cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, 1, dropout=dropout, bidirectional=self.bidi)

        elif cell_type == "RNN":
            self.rnn = nn.RNN(embedding_size, hidden_size, 1, dropout=dropout, bidirectional=self.bidi)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x_trai):

        embedding = self.dropout(self.embedding(x_trai))
        if isinstance(self.rnn, nn.LSTM):
            enc_states, (hidden, cell) = self.rnn(embedding)

            hidden = self.fc_hidden(
                torch.cat((hidden[0:1], hidden[1:2]), dim = 2)
            )
            cell = self.fc_cell(
                torch.cat((cell[0:1], cell[1:2]), dim = 2)
            )
            return enc_states, hidden, cell

        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            enc_states, hidden = self.rnn(embedding)
            hidden = self.fc_hidden(
                torch.cat((hidden[0:1], hidden[1:2]), dim = 2)
            )
            return enc_states, hidden

class EnteDecoder(nn.Module):

    def __init__(self, cell_type: str, input_size, embedding_size, hidden_size, output_size, dropout):

        super(EnteDecoder, self).__init__()

        self.bidi = False
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(input_size, embedding_size)

        assert cell_type is not None, "Provide a valid cell type"

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size + hidden_size * 2, hidden_size, 1, dropout=dropout, bidirectional=self.bidi)
        
        elif cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size + hidden_size * 2, hidden_size, 1, dropout=dropout, bidirectional=self.bidi)

        elif cell_type == "RNN":
            self.rnn = nn.RNN(embedding_size + hidden_size * 2, hidden_size, 1, dropout=dropout, bidirectional=self.bidi)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim = 0)
        self.relu = nn.ReLU()

        if self.bidi:
            self.fully_conn = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fully_conn = nn.Linear(hidden_size, output_size)

    def forward(self, x_trai, encoder_states, hidden, cell):

        x_trai = x_trai.unsqueeze(0)
        embedding = self.dropout(self.embedding(x_trai))

        seq_len = encoder_states.shape[0]
        h_reshaped = hidden.repeat(seq_len, 1, 1)

        energy = self.relu(
            self.energy(
                torch.cat(
                    (h_reshaped, encoder_states), dim=2
                )
            )
        )

        attention = self.softmax(energy).permute(1,0,2)
        encoder_states = encoder_states.permute(1,0,2)

        c_vect = torch.bmm(attention.permute(0,2,1), encoder_states).permute(1,0,2)

        inp = torch.cat((c_vect, embedding), dim=2)

        if isinstance(self.rnn, nn.LSTM):
            output, (hidden, cell) = self.rnn(inp, (hidden, cell))

        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            output, hidden = self.rnn(inp, hidden)

        predictions = self.fully_conn(output)
        predictions = torch.softmax(predictions, dim = 2)
        predictions = predictions.squeeze(0)

        if isinstance(self.rnn, nn.LSTM):
            return predictions, hidden, cell
        elif isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            return predictions, hidden

class EnteSeq2SeqAttn(nn.Module):


    # Reference: https://www.youtube.com/watch?v=sQUqQddQtB4
    # The implementation has been done by referring the above video 
    # for inspiration. Code hasn't been copied directly but there might
    # be similarities. 

    def __init__(self, encoder, decoder, device):

        super(EnteSeq2SeqAttn, self).__init__()
        self.encoder: EnteEncoder = encoder
        self.decoder: EnteDecoder = decoder
        self.device = device
        self.bidi = True

    def forward(self, source, target, tfr = 0.5):

        batch_siz = source.shape[1]
        target_len = target.shape[0]
        target_vocab_length = train_data.num_decoder_tokens #TODO total number of letters in malayalam

        outputs = torch.zeros(target_len, batch_siz, target_vocab_length).to(self.device)

        if isinstance(self.encoder.rnn, nn.GRU) or isinstance(self.encoder.rnn, nn.RNN):
            enc_states, hidden = self.encoder(source)

        if isinstance(self.encoder.rnn, nn.LSTM):
            enc_states, hidden, cell = self.encoder(source)

        x_targ = target[0]

        for count in range(1, target_len):

            if isinstance(self.decoder.rnn, nn.LSTM):
                out, hidden, cell = self.decoder(x_targ, enc_states, hidden, cell)
            elif isinstance(self.decoder.rnn, nn.GRU) or isinstance(self.encoder.rnn, nn.RNN):
                out, hidden = self.decoder(x_targ, enc_states,  hidden, None)

            outputs[count] = out
            best = out.argmax(1)
            x_targ = target[count] if random.random() < tfr else best

        return outputs

class EnteTransliteratorAttn():
    def __init__(
        self,
        cell_type: str = "LSTM",
        epochs: int = 20,
        hidden_size: int = 128,
        lr: float = 1e-3,
        batch_size: int = 128,
        dropout: float = 0.5,
        emb: int = 100,
        search_method: str = 'beam',
        beam_width: int = 4
    ):
        self.cell_type = cell_type
        self.epochs = epochs

        self.encoder_layers = 1
        self.decoder_layers = 1
        self.hidden_size = hidden_size
        self.learning_rate = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.bidirectional = True
        self.embedding_size = emb
        self.search_method = search_method
        self.beam_width = beam_width

        self.model: EnteSeq2SeqAttn = None #TODO Optional
        self.inference_mode = False


    def inference_model(self):

        self.model.eval()
        self.model.encoder.eval()
        self.model.decoder.eval()

        return self.model.encoder, self.model.decoder

    def __greedy_search(self, input_sequence, data):

        encoder, decoder = self.inference_model()
        with torch.no_grad():
            encoder_cell = None
            if isinstance(encoder.rnn, nn.GRU) or isinstance(encoder.rnn, nn.RNN):
                encoder_hidden = encoder(input_sequence)

            if isinstance(encoder.rnn, nn.LSTM):
                encoder_hidden, encoder_cell = encoder(input_sequence)

            start_token_index = data.target_chars_index[Data.start]
            end_token_index = data.target_chars_index[Data.end]

            sequence = torch.tensor([[start_token_index]]).to(DEVICE)
            hidden, cell = encoder_hidden, encoder_cell

            for _ in range(data.max_decoder_input_length):

                if isinstance(decoder.rnn, nn.LSTM):
                    out, hidden, cell = decoder(sequence[:, -1], hidden, cell)

                if isinstance(decoder.rnn, nn.GRU) or isinstance(decoder.rnn, nn.RNN):
                    out, hidden = decoder(sequence[:, -1], hidden, None)

                probabilities = torch.softmax(out, dim=-1)
                _, top_token = torch.max(probabilities, dim=-1)

                current_token = top_token.unsqueeze(1)
                sequence = torch.cat((sequence, current_token), dim=1)

                if current_token.item() == end_token_index:
                    break

            token_indices = sequence.squeeze().tolist()

            return token_indices

    def __beam_search(self, inp, data, beam_width):
        
        encoder, decoder = self.inference_model()
        with torch.no_grad():

            encoder_cell = None
            if isinstance(encoder.rnn, nn.GRU) or isinstance(encoder.rnn, nn.RNN):
                encoder_hidden = encoder(inp)

            if isinstance(encoder.rnn, nn.LSTM):
                encoder_hidden, encoder_cell = encoder(inp)
            
            start_token_index = data.target_chars_index[Data.start]
            current_beam = [(torch.tensor([[start_token_index]]).to(DEVICE), 0.0, encoder_hidden, encoder_cell)]

            completed_sequences = []

            for _ in range(data.max_decoder_input_length):

                new_beam = []

                for sequence, sequence_score, hidden, cell in current_beam:

                    last_token = sequence[:, -1]

                    if isinstance(decoder.rnn, nn.LSTM):
                        out, hidden, cell = decoder(last_token, hidden, cell)

                    if isinstance(decoder.rnn, nn.GRU) or isinstance(decoder.rnn, nn.RNN):
                        out, hidden = decoder(last_token, hidden, None)

                    log_probs = torch.log_softmax(out, dim = -1)

                    topk_log_probs, topk_tokens = log_probs.topk(beam_width, dim = -1)

                    for i in range(beam_width):

                        token = topk_tokens[:,i]
                        token_score = topk_log_probs[:,i]

                        new_seq = torch.cat((sequence, token.unsqueeze(0)), dim = 1)
                        new_score = sequence_score + token_score.item()

                        if token.item() == data.target_chars_index[Data.end]:
                            completed_sequences.append((new_seq, new_score))
                        else:
                            new_beam.append((new_seq, new_score, hidden, cell))

                new_beam.sort(key=lambda x: x[1], reverse = True)
                current_beam = new_beam[:beam_width]

                if all(sequence[:, -1].item() == data.target_chars_index[Data.end] for sequence, _, _, _ in current_beam):
                    break

            completed_sequences.sort(key=lambda x: x[1], reverse=True)

            if not completed_sequences:
                return self.__greedy_search(inp, data)

            best_sequence = completed_sequences[0][0]

            # Convert the best sequence to a list of token indices
            token_indices = best_sequence.squeeze().tolist()

            return token_indices

            

    def decode_sequence(self, input_sequence, data, search_method = 'beam', beam_width = 4):
        

        if search_method == 'greedy':
            return self.__greedy_search(input_sequence, data)
        elif search_method == 'beam':
            return self.__beam_search(input_sequence, data, beam_width)
        else:
            raise ValueError("Invalid search method. Please choose 'greedy' or 'beam'.")


    def test(self, data: Data):

        total = 0
        correct = 0
        for i in range(len(data.source)):

            input_sequence, target_sequence = data.get_data_point(i, DEVICE)

            decoded_tokens = self.decode_sequence(
                input_sequence, 
                data,
                search_method=self.search_method, 
                beam_width=self.beam_width
            )
            decoded_word = data.indices_to_word(decoded_tokens)
            target_word = data.indices_to_word(list(itertools.chain(*target_sequence.tolist()))) #TODO

            total += 1
            if decoded_word == target_word:
                correct += 1


        accuracy = correct / total
        self.model.train()
        self.model.encoder.train()
        self.model.decoder.train()

        return accuracy

    def train(self):
    
        train_data.set_batch_size(self.batch_size)
        valid_data.set_batch_size(self.batch_size)
        input_size_encoder = train_data.num_encoder_tokens
        input_size_decoder = train_data.num_decoder_tokens

        total_batches = len(train_data.source) // self.batch_size

        assert self.model is not None

        encoder = EnteEncoder(
            self.cell_type,
            input_size_encoder,
            self.embedding_size,
            self.hidden_size,
            self.dropout
        )

        decoder = EnteDecoder(
            self.cell_type,
            input_size_decoder,
            self.embedding_size,
            self.hidden_size,
            input_size_decoder,
            self.dropout
        )

        self.model = EnteSeq2SeqAttn(encoder, decoder, DEVICE).to(DEVICE)

        loss = nn.CrossEntropyLoss(ignore_index=train_data.target_chars_index[Data.pad])
        optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

        los = None
        val_acc = 0

        for epoch in range(self.epochs):

            total_correct = 0
            total_samples = 0

            self.model.train()
            assert not self.inference_mode

            print(f'Epoch {epoch + 1} of {self.epochs} at {time.datetime.now()}')

            # for __ in range(1): #TODO
            for __ in range(total_batches):

                input_data, target = train_data.get_batch(DEVICE)
                output = self.model(input_data, target).to(DEVICE)

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
    
            sample_in, sample_target = train_data.get_random_sample(0)
            sample_in = sample_in.to(DEVICE)
            sample_target = sample_target.to(DEVICE)
            pred = self.model(
                sample_in.unsqueeze(1),
                sample_target.unsqueeze(1)
            )

        val_acc = self.test(valid_data)

        wb.log({"train_loss": los})
        wb.log({"validation_accuracy": val_acc})


attn = EnteTransliteratorAttn(
    cell_type = "RNN",
    epochs = 20,
    hidden_size=128,
    lr=1e-3,
    batch_size=128,
    dropout=0.3,
    emb=150,
    search_method='beam',
    beam_width=4
)

attn.train()