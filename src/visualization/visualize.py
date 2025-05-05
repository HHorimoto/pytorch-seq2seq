import torch

import numpy as np
import matplotlib.pyplot as plt

def plot(results: dict, metric: str):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel(metric)
    for key, value in results.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.savefig(metric+'.png')