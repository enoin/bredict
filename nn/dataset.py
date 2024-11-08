import numpy as np
import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from util.seq import z_norm, min_max_scale, sma, cma, ema


def zip_series_collate(batch):
    data, next_seq, idx = zip(*batch)
    next_seq = torch.stack(next_seq, dim=0).unsqueeze(dim=2)
    st = torch.stack([torch.stack(list_of_tensors, dim=1) for list_of_tensors in data], dim=0)
    return st, next_seq, idx


def read_csv():
    df = pandas.read_csv('dataset/BTC-2021min.csv',
                         index_col=None,
                         header=0, nrows=3000)
    uc = df[['unix', 'close', 'high']]
    uc.sort_values(by='unix', ascending=True, inplace=True)
    print(df)
    return uc


class TsDataset(Dataset):

    def __init__(self, shift_len=4, seq_length=50):
        self.data = read_csv()
        self.cv = self.data[['unix', 'close', 'high']]
        self.values = z_norm(torch.tensor(self.cv['close'], dtype=torch.float32))
        self.values = min_max_scale(torch.tensor(self.cv['close'], dtype=torch.float32))
        self.s = min_max_scale(torch.tensor(sma(self.values, 4), dtype=torch.float32))
        self.c = min_max_scale(torch.tensor(cma(self.values), dtype=torch.float32))
        self.e = min_max_scale(torch.tensor(ema(self.values, 0.2), dtype=torch.float32))

        self.seq_length = seq_length
        self.shift_len = shift_len
        self.data_len = self.data.__len__()

    def __getitem__(self, idx):
        sequence_end = idx + self.seq_length
        next_sequence_begin = idx + self.seq_length + 1
        next_sequence_end = next_sequence_begin + self.shift_len
        sequence = self.values[idx: sequence_end]
        next_sequence = self.values[next_sequence_begin: next_sequence_end]

        s = self.s[idx: sequence_end]
        c = self.c[idx: sequence_end]
        e = self.e[idx: sequence_end]

        return [sequence, s, c, e], next_sequence, idx

    def __len__(self):
        return self.data_len - (self.seq_length + self.shift_len + 1)

    @staticmethod
    def get_data_loader(batch_size, workers):
        dataset = TsDataset()

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            drop_last=True,
            collate_fn=zip_series_collate
        )
