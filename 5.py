import csv
import math
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py
import nn

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=' ')
        data = list(lines)
        for i in range(len(data)):
            point = list(map(float, data[i]))
            data[i] = point
        return data

def main():
    train_ds = import_dataset("sincTrain25.dt")
    test_ds = import_dataset("sincValidate10.dt")
    # ds = import_dataset("sinc10.dt")

    tr_x = [[i[0]] for i in train_ds]
    tr_y = [[i[1]] for i in train_ds]

    te_x = [[i[0]] for i in test_ds]
    te_y = [[i[1]] for i in test_ds]

    # print(x)
    # print(y)

    nn.nn(tr_x, tr_y, te_x, te_y)


if __name__ == '__main__':
    main()
