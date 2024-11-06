from torch import nn, cat
import torch.nn.functional as F


class BredictNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 100
        self.hidden_dim = 100
        self.n_layers = 100
        self.output_size = 3

        self.rnn = nn.RNN(self.input_size, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

        # self.output = nn.RNN(3, 3, kernel_size=3, stride=1, padding=1)
        # self.output = nn.Linear(3, 3, kernel_size=3, stride=1, padding=1)
        # self.output = nn.GRUCell(3, 3, kernel_size=3, stride=1, padding=1)
        # self.output = nn.RNNCell(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, hidden=None):

        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        # r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)

        return output, hidden
