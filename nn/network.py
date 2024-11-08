import torch
from torch import nn, cat


class BredictNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_input_size = 4
        self.lstm_hidden_size = 64
        self.lstm_input_size = self.rnn_hidden_size = 64
        self.rnn_n_layers = 8
        self.lstm_n_layers = 8
        self.output_size = 4
        self.batch_size = 4

        self.rnn = nn.RNN(self.rnn_input_size, self.rnn_hidden_size, self.rnn_n_layers, batch_first=True, dropout=0.2)
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_n_layers,
                            batch_first=True, dropout=0.1)

        self.fc = nn.Linear(self.lstm_hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, self.output_size)

    def forward(self, x, hidden, lstm_hidden):
        r_out, h_n = self.rnn(x, hidden)
        lstm_out, lstm_hidden = self.lstm(r_out, lstm_hidden)
        lhs = lstm_out[:, -1, :]
        fc = self.fc(lhs)
        fc = self.relu(fc)
        output = self.fc2(fc)
        return output.unsqueeze(2), h_n, lstm_hidden

    def init_hidden(self):
        hidden = torch.zeros(self.rnn_n_layers, self.batch_size, self.rnn_hidden_size).to(self.device)
        return hidden

    def lstm_hidden(self):
        return (torch.zeros(self.lstm_n_layers, self.batch_size, self.lstm_hidden_size).to(self.device),
                torch.zeros(self.lstm_n_layers, self.batch_size, self.lstm_hidden_size).to(self.device))

    @property
    def device(self):
        return next(self.parameters()).device
