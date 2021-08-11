import numpy as np
import pandas as pd


def calculate_cell_correlation(df, frame_index, size=9):
    sliding_window = df.iloc[frame_index:frame_index + size, :]
    return sliding_window.corr()


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


def calculate_quantiles(quantile_value, df, columns=None) -> pd.Series:
    if columns is None:
        columns = df.columns
    quantile_series = pd.Series(index=columns)

    for index, values in df.iteritems():
        quantile_series[index] = np.quantile(values, quantile_value)

    return quantile_series
