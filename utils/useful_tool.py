import numpy as np
import pandas as pd


def find_all_nan_cells(matrix: pd.DataFrame) -> list:
    nan_cells = []
    for cell_name, items in matrix.iteritems():
        if items.isnull().all():
            nan_cells.append(cell_name)

    return nan_cells


def dropped_unique_cells(cells1, cells2):
    dropped_cells = np.array([])
    dropped_cells = np.append(dropped_cells, cells1)
    dropped_cells = np.append(dropped_cells, cells2)
    dropped_cells = np.unique(dropped_cells)
    return dropped_cells


def calc_event_rate(spikes, second, threshold=0.0):
    return len(spikes[spikes > threshold]) / second
