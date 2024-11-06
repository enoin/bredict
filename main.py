import time

from matplotlib import pyplot as plt

from nn.dataset import read_csv
from nn.runner import NNRunner
from util.device import check_mps_devices
import csv
import numpy as np
import torch

from util.plot import show_series, SeriesPlotter

check_mps_devices()

# cv = read_csv()
# values = cv['close'].tolist()
# show_series(values,[3112, 1322, 9232, 12324, 15, 13, 322317, 19, 123228, 20] )

# sp = SeriesPlotter(False)
# sp.set_data(12, 2)
# sp.set_data(12, 2)
# sp.set_data(12, 2)
# sp.show()

runner = NNRunner()
runner.train()
