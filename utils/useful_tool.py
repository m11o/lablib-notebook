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


def is_all_zeros_constant_array(array):
    return np.all(array == 0.0)

def convert_polar_df(df, xy_df):
    body_parts = df.columns.get_level_values(0).unique(0).to_list()
    iterables = [body_parts, ['distance', 'theta']]
    columns = pd.MultiIndex.from_product(iterables, names=['bodyparts', 'coords'])
    polar_df = pd.DataFrame(columns=columns)

    centroid_x = xy_df.loc[:, 'X']
    centroid_y = xy_df.loc[:, 'Y']
    distance_by_frame = xy_df.loc[:, 'Distance']

    for body_part in body_parts:
        x = df.loc[:, (body_part, 'x')]
        y = df.loc[:, (body_part, 'y')]

        diff_x = x - centroid_x
        diff_y = y - centroid_y
        distance = np.sqrt(diff_x**2 + diff_y**2)
        theta = np.arctan2(diff_y, diff_x)

        polar_df.loc[:, (body_part, 'distance')] = distance
        polar_df.loc[:, (body_part, 'theta')] = theta
    polar_df[(None, 'distance_by_frame')] = distance_by_frame

    return polar_df
