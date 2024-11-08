import time
import numpy as np
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
        self.predicted2 = []
        self.predicted3 = []
        self.predicted4 = []
        self.fig, self.ax = plt.subplots(figsize=(15, 5))
        self.text = self.fig.text(0.95, 0.5, f"Batch: 0\n loss: -", ha='center', va='center', fontsize=12,
                                  bbox=dict(facecolor='white', edgecolor='black'))

        self.line_predict, = self.ax.plot([], label="predicted")
        self.line_predict_2, = self.ax.plot([], label="predicted")
        self.line_actual, = self.ax.plot([], linewidth=0.6, label='actual', color="r")
        self.line_c, = self.ax.plot([], linewidth=0.6, label='c', color="g")

    def set_base_line(self, dataset):
        actual = dataset.values
        c = dataset.c
        self.line_actual.set_ydata(actual)
        self.line_actual.set_xdata(np.arange(len(actual)))

        self.line_c.set_ydata(c)
        self.line_c.set_xdata(np.arange(len(c)))

        self.predicted = actual[:50].numpy()
        self.update()

    def set_prediction(self, predicted, batch, loss, index):
        self.text.set_text(f"Batch: {batch}\n loss: {loss:>9f}")
        p = predicted.clone().detach().squeeze(-1)
        self.predicted = np.concatenate((self.predicted,  p[0].cpu().flatten()))
        self.line_predict.set_ydata(self.predicted)
        self.line_predict.set_xdata(np.arange(len(self.predicted)))

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(3.1)
        time.sleep(3.1)

    def show(self):
        plt.grid(True)
        if self.nonblocking:
            plt.ion()
        plt.show()
