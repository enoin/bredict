import torch
from torch import nn, cat
import torch.nn.functional as F


class BredictNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 4
        self.hidden_dim = 4
        self.n_layers = 8
        self.output_size = 1
        self.batch_size = 4

        self.rnn = nn.RNN(self.input_size, self.hidden_dim, self.n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(self.hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, self.output_size)  # Second fully connected layer

        # self.output = nn.LSTM(3, 3, kernel_size=3, stride=1, padding=1)
        # self.output = nn.RNN(3, 3, kernel_size=3, stride=1, padding=1)
        # self.output = nn.Linear(3, 3, kernel_size=3, stride=1, padding=1)
        # self.output = nn.GRUCell(3, 3, kernel_size=3, stride=1, padding=1)
        # self.output = nn.RNNCell(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, hidden):
        r_out, h_n = self.rnn(x, hidden)
        last_hidden_state = r_out[:, -1, :]
        lh_ = h_n[-1]
        # output = self.fc(last_hidden_state)
        x = self.fc(last_hidden_state)
        x = self.relu(x)
        output = self.fc2(x)

        # output = self.fc(last_hidden_state).unsqueeze(2)
        return output, h_n

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        return hidden
