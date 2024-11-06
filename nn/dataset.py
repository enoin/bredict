import numpy as np
import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from util.seq import z_norm, min_max_scale


def zip_series_collate(batch):
    data, next, idx = zip(*batch)
    return torch.stack(data), torch.stack(next), idx


def read_csv():
    df = pandas.read_csv('dataset/BTC-2021min.csv',
                         index_col=None,
                         header=0)
    uc = df[['unix', 'close']]
    uc.sort_values(by='unix', ascending=True, inplace=True)
    print(df)
    return uc


class TsDataset(Dataset):

    def __init__(self, shift_len=3, seq_length=100):
        self.data = read_csv()
        self.cv = self.data[['unix', 'close']]
        # self.values = z_norm(torch.tensor(self.cv['close'], dtype=torch.float32))
        self.values = min_max_scale(torch.tensor(self.cv['close'], dtype=torch.float32))
        self.seq_length = seq_length
        self.shift_len = shift_len
        self.data_len = self.data.__len__()

    def __getitem__(self, idx):
        sequence_end = idx + self.seq_length
        next_sequence_begin = idx + self.seq_length + 1
        next_sequence_end = next_sequence_begin + self.shift_len
        sequence = self.values[idx: sequence_end]
        next_sequence = self.values[next_sequence_begin: next_sequence_end]
        return sequence, next_sequence, idx

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
