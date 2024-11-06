import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt


def show_series(seq1, seq2):
    plt.figure(figsize=(15, 5))
    plt.text(2, 3, 'This is a Matplotlib text box\nfrom how2matplotlib.com',
            bbox=dict(facecolor='white', edgecolor='black'))
    plt.plot(seq1, color='b')
    plt.plot(seq2, color='r')
    plt.xlabel("T")
    plt.ylabel("P")
    plt.grid(True)
    plt.show()


class SeriesPlotter:

    def __init__(self, nonblocking=True):
        self.nonblocking = nonblocking
        self.predicted = []
        self.fig, self.ax = plt.subplots(figsize=(15, 5))
        self.text = self.fig.text(0.95, 0.5, f"Batch: 0\n loss: -", ha='center', va='center', fontsize=12,
                      bbox=dict(facecolor='white', edgecolor='black'))

        self.line_predict, = self.ax.plot([], label="predicted")
        self.line_actual, = self.ax.plot([], linewidth=0.6,  label='actual', color="r")
        self.fill = self.ax.fill_between([], [], [], alpha=.5, linewidth=0)


    def set_base_line(self, series):
        self.line_actual.set_ydata(series)
        self.line_actual.set_xdata(np.arange(len(series)))
        self.predicted = series[:10]
        self.update()

    def set_prediction(self, predicted, batch, loss):
        self.text.set_text(f"Batch: {batch}\n loss: {loss:>9f}")
        # random_floats = [random.uniform(0.316, 0.319) for _ in range(200)]
        # random_floats2 = [random.uniform(0.314, 0.316) for _ in range(200)]
        # self.fill.remove()
        # self.fill = self.ax.fill_between(np.arange(len(random_floats)), random_floats, random_floats2, color='blue', alpha=0.3)  # Add updated fill
        self.predicted = np.concatenate((self.predicted, predicted.detach().numpy()))
        self.line_predict.set_ydata(self.predicted)
        self.line_predict.set_xdata(np.arange(len(self.predicted)))

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)
        time.sleep(0.01)

    def show(self):
        plt.grid(True)
        if self.nonblocking:
            plt.ion()
        plt.show()

