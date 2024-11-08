import torch
from torch import nn
from nn.dataset import TsDataset
from nn.network import BredictNetwork
from util.device import get_device
from util.plot import SeriesPlotter

SEPARATE = "------------------------------"
MODEL_NAME = 'predict_model.pth'


class NNRunner:
    learning_rate = 0.08
    batch_size = 4
    epochs = 4

    def __init__(self):
        self._device = get_device()
        self.sp = SeriesPlotter()
        self.sp.show()

    def train_loop(self, dataloader, model, loss_fn, optimizer):
        model.train()
        size = len(dataloader.dataset)
        running_loss = 0

        hidden = model.init_hidden()
        lstm_hidden = model.lstm_hidden()

        for batch, (data, next_values, index) in enumerate(dataloader, 0):
            data, next_values = data.to(self._device), next_values.to(self._device)
            predicted, hidden, lstm_hidden = model(data, hidden, lstm_hidden)
            loss = loss_fn(predicted, next_values)

            hidden = hidden.detach()
            lstm_hidden = tuple([h.detach() for h in lstm_hidden])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            self.sp.set_prediction(predicted, batch, loss, index)

            if batch % 40 == 0:
                loss, current = loss.item(), (batch + 1) * len(data)
                print(f"batch: [{batch}] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                self.sp.update()

        return running_loss

    def train(self):
        model = BredictNetwork()
        model = model.to(self._device)
        print(model)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        loader = TsDataset.get_data_loader(self.batch_size, 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss = nn.MSELoss()

        self.sp.set_base_line(loader.dataset)

        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n" + SEPARATE)
            self.train_loop(loader, model, loss, optimizer)

        torch.save(model, MODEL_NAME)
        print("Done!")

    def load_model(self):
        model = torch.load(MODEL_NAME, weights_only=False)
        model = model.to(self._device)
        model.eval()
        return model

    def run(self):
        model = self.load_model()
        print(model)
