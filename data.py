"""
Assignment - 3

@author: CS22M056
"""

import random
import numpy as np
import torch

from aux import load_data

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Data:
    """TODO"""

    # pad = "#"
    pad = " "
    # start = "@"
    start = "I"
    # end = "*"
    end = "F"

    def __init__(self, path: str):


        self.source, self.target = load_data(path)
        self.source_dict = []
        self.target_dict = []
        self.source_chars = []
        self.target_chars = []

        self.batch_size : int = None

        self.max_encoder_input_length = None
        self.max_decoder_input_length = None
        self.num_encoder_tokens = None
        self.num_decoder_tokens = None

        self.source_chars_index = None
        self.target_chars_index = None
        self.rev_source_chars_index = None
        self.rev_target_chars_index = None

        self.__generate_vocab(self.source, self.target)

        self.encoder_source_data = np.zeros(
            (len(self.source_dict), self.max_encoder_input_length), dtype="float32"
        )

        self.decoder_source_data = np.zeros(
            (len(self.source_dict), self.max_decoder_input_length), dtype="float32"
        )
        self.decoder_target_data = np.zeros(
            (
                len(self.source_dict),
                self.max_decoder_input_length,
                self.num_decoder_tokens
            ), dtype="float32"
        )

        self.__generate_input()

    def __generate_vocab(self, source, target):
        """TODO"""

        source_chars = set()
        target_chars = set()

        for (source_word, target_word) in zip(source, target):

            target_word = "I" + target_word + "F"
            self.source_dict.append(source_word)
            self.target_dict.append(target_word)

            for char in source_word:
                source_chars.add(char)

            for char in target_word:
                target_chars.add(char)

        random = np.arange(len(source))
        np.random.shuffle(random)

        source_temp = []
        target_temp = []

        for i in range(len(source)):

            source_temp.append(self.source_dict[random[i]])
            target_temp.append(self.target_dict[random[i]])

        self.source_dict = source_temp
        self.target_dict = target_temp

        source_chars.add(" ")
        target_chars.add(" ")

        self.source_chars = sorted(list(source_chars))
        self.target_chars = sorted(list(target_chars))

        self.source_chars_index = dict([(char, i) for i, char in enumerate(self.source_chars)])
        self.target_chars_index = dict([(char, i) for i, char in enumerate(self.target_chars)])

        self.max_encoder_input_length = max([len(word) for word in self.source_dict])
        self.max_decoder_input_length = max([len(word) for word in self.target_dict])
        self.num_encoder_tokens = len(self.source_chars)
        self.num_decoder_tokens = len(self.target_chars)

    def __generate_input(self):
        """TODO"""

        for i, (source_word, target_word) in enumerate(zip(self.source_dict, self.target_dict)):

            pos = 0
            for pos, char in enumerate(source_word):
                self.encoder_source_data[i, pos] = self.source_chars_index[char]
            self.encoder_source_data[i, pos+1:] = self.source_chars_index[" "]

            for pos, char in enumerate(target_word):

                self.decoder_source_data[i, pos] = self.target_chars_index[char]
                if pos > 0:
                    self.decoder_target_data[i, pos - 1, self.target_chars_index[char]] = 1.0
            self.decoder_source_data[i, pos+1:] = self.target_chars_index[" "]
            self.decoder_target_data[i, pos:, self.target_chars_index[" "]] = 1.0

        self.rev_source_chars_index = dict((i, char) for char, i in self.source_chars_index.items())
        self.rev_target_chars_index = dict((i, char) for char, i in self.target_chars_index.items())

    def get_batch(self, device):
        assert self.batch_size is not None, "Set batch size first"
        
        # Randomly select indices for the batch
        indices = np.random.randint(0, len(self.source_dict), size=self.batch_size)
        
        # Retrieve the corresponding input and target data
        input_data = self.encoder_source_data[indices]
        target_data = self.decoder_source_data[indices]
        # Convert the numpy arrays to PyTorch tensors
        input_data = torch.tensor(input_data, dtype=torch.long).transpose(0, 1).to(device)
        target_data = torch.tensor(target_data, dtype=torch.long).transpose(0, 1).to(device)
        
        # print(input_data.shape, target_data.shape)
        return input_data, target_data

    def get_data_point(self, index, device):

        input_data = self.encoder_source_data[index]
        target_data = self.decoder_source_data[index]

        # Convert the numpy arrays to PyTorch tensors
        input_data = torch.tensor(input_data, dtype=torch.long).unsqueeze(1).to(device)
        target_data = torch.tensor(target_data, dtype=torch.long).unsqueeze(1).to(device)
        
        # print(input_data.shape)
        return input_data, target_data


    def set_batch_size(self, size):
        self.batch_size = size


    def indices_to_word(self, indices, source=False):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if not isinstance(indices, list):
            indices = [indices]

        words = []
        for idx in indices:
            if idx == self.target_chars_index[self.end] or (source and idx == self.source_chars_index[" "]):
                break
            char = self.rev_target_chars_index[idx]
            if source:
                char = self.rev_source_chars_index[idx]
            words.append(char)

        word = "".join(words)
        return word[1:]

    def get_random_sample(self, index = None):

        if index is None:
            index = random.randint(0, len(self.source_dict) - 1)
        input_sequence = self.encoder_source_data[index]
        target_sequence = self.decoder_source_data[index]
        input_data = torch.tensor(input_sequence, dtype=torch.long)
        target = torch.tensor(target_sequence, dtype=torch.long)
        return input_data, target

    def sequence_to_text(self, sequence, source = False):

        # print(len(self.source), self.rev_target_chars_index, "\n\n")
        # print(len(self.rev_target_chars_index), self.rev_target_chars_index[70])
        if sequence.dim() == 0:
            sequence = sequence.unsqueeze(0)
        text = ""
        for idx in sequence:
            char = self.rev_target_chars_index[idx.item()]
            if source:
                char = self.rev_source_chars_index[idx.item()]
            if char == self.end or (source and char == " "):
                break
            text += char
        return text

