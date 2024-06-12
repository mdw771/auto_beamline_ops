import io

import pandas as pd
import numpy as np


def read_nor(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        columns = []
        data = []
        for l in lines:
            if 'norm' in l and ' e ' in l:
                columns = l[1:].strip().split()
            if not l.startswith('#'):
                data.append([float(x) for x in l.strip().split()])
    data = np.array(data)
    table = pd.DataFrame({columns[i]: data[:, i] for i in range(len(columns))})
    return table
