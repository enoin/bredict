import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def z_norm(tensor):
    mean, std = tensor.mean(dim=-1, keepdim=True), tensor.std(dim=-1, keepdim=True)
    normed_seqs = (tensor - mean) / std
    return normed_seqs


def normalize_range(tensor, a=0, b=1):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = a + (tensor - min_val) * (b - a) / (max_val - min_val)
    return normalized_tensor


def min_max_scale(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def sma(seq, window):
    """ simple moving average """
    moving_avg = np.convolve(seq, np.ones(window) / window, mode='valid')
    return moving_avg


def cma(seq):
    cumulative_avg = np.cumsum(seq) / np.arange(1, len(seq) + 1)
    return cumulative_avg


def ema(seq, alpha):
    """ Exponential Moving Average """
    exp_ma = np.zeros_like(seq)
    exp_ma[0] = seq[0]
    for t in range(1, len(seq)):
        exp_ma[t] = alpha * seq[t] + (1 - alpha) * exp_ma[t - 1]
    return exp_ma


def error_rmse(test_seq, pred_seq):
    mse = mean_squared_error(test_seq, pred_seq)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    mae = mean_absolute_error(test_seq, pred_seq)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    mape = mean_absolute_percentage_error(test_seq, pred_seq) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
