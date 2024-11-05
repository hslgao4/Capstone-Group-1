import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from torch.utils.data import DataLoader, TensorDataset

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Encoder LSTM
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(x)

        # Attention mechanism
        h_n_repeated = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        attention_scores = torch.tanh(self.attention(torch.cat((encoder_outputs, h_n_repeated), dim=2)))
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)

        # Decoder LSTM
        decoder_input = context_vector.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_output, _ = self.decoder_lstm(decoder_input, (h_n, c_n))

        # Only take the last output of the decoder for each sequence
        final_output = self.fc(decoder_output[:, -1, :])  # shape: (batch_size, output_size)
        return final_output

