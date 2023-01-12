import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(data, pred, model):
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data.temp)
    if model == "AR" or model == "Both":
        ax.plot(pred)
    elif model == "DeepAR" or model == "Both":
        total_fc_dar = np.zeros(36)
        for i in range(len(pred[0].samples)):
            total_fc_dar += pred[0].samples[i]
        total_fc_dar = total_fc_dar/len(pred[0].samples)
        pred_dar = pd.DataFrame(data=total_fc_dar, index=data.index[-36:])
        ax.plot(pred_dar)
    return fig


def final_plot(data, pred_ar, pred_deepar):
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data.temp[-200:])
    ax.plot(pred_ar)
    total_fc_dar = np.zeros(36)
    for i in range(len(pred_deepar[0].samples)):
        total_fc_dar += pred_deepar[0].samples[i]
    total_fc_dar = total_fc_dar / len(pred_deepar[0].samples)
    pred_dar = pd.DataFrame(data=total_fc_dar, index=data.index[-36:])
    ax.plot(pred_dar)
    return fig
