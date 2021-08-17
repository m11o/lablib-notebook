import numpy as np
import pandas as pd

from math import sqrt


def calculate_cell_correlation(df, frame_index, size=9):
    sliding_window = df.iloc[frame_index:frame_index + size, :]
    corr = sliding_window.corr()
    corr.fillna(0.0, inplace=True)
    return corr


def build_cell_correlation_df(df, start, end):
    seconds = end - start
    cell_correlations_df = pd.DataFrame(
        columns=list(range(len(df.columns) ** 2)),
        index=list(range(seconds))
    )
    for index, second in enumerate(range(start, end)):
        frame_index = second * 10
        corr_df = calculate_cell_correlation(df, frame_index)

        corr_df = corr_df - (corr_df * np.eye(len(corr_df)))
        cell_correlations_df.iloc[index, :] = corr_df.to_numpy().ravel()

    return cell_correlations_df


def calculate_time_correlation(seconds, time_correlation_df, cell_correlation_dfs):
    for i in range(seconds):
        i_time_data = cell_correlation_dfs[i]

        for j in range(seconds):
            j_time_data = cell_correlation_dfs[j]

            if time_correlation_df.at[j, i] is not np.nan:
                time_correlation_df.at[i, j] = time_correlation_df.at[j, i]
                continue

            cell_count = len(i_time_data)
            calculated_values = i_time_data * j_time_data
            corr_sum = (calculated_values - (calculated_values * np.eye(len(calculated_values)))).sum().sum()
            if time_correlation_df.at[i, j] is np.nan:
                time_correlation_df.at[i, j] = 0.0
            time_correlation_df.at[i, j] += corr_sum / (cell_count * (cell_count - 1))


def calculate_shuffle_time_correlation(cell_correlation_df):
    cell_size = sqrt(len(cell_correlation_df.columns))
    return (np.dot(cell_correlation_df, cell_correlation_df.T)) / (cell_size * (cell_size - 1))


def calculate_quantiles(quantile_value, df, columns=None) -> pd.Series:
    if columns is None:
        columns = df.columns
    quantile_series = pd.Series(index=columns)

    for index, values in df.iteritems():
        quantile_series[index] = np.quantile(values, quantile_value)

    return quantile_series
