import pandas as pd
import numpy as np

from time_correlation_tool import calculate_time_correlation, calculate_cell_correlation


class ShuffleTimeCorrelationCalculator:
    SHUFFLE_COUNT = 1000

    def __init__(self, all_df, engram_df, start, end):
        self.all_df = all_df
        self.engram_df = engram_df
        self.start = start
        self.end = end
        self.seconds = self.end - self.start

        self.average_variation = self.__shuffle_default_df()
        self.abs_sum_variation = self.__shuffle_default_df()

    def calc(self):
        for n in range(self.SHUFFLE_COUNT):
            random_cells = np.random.randint(0, len(self.all_df.columns), size=len(self.engram_df.columns))
            shuffled_cell_df = self.all_df.iloc[:, random_cells].copy(deep=True)

            self.__calculate_shuffled(n, shuffled_cell_df, self.average_variation, self.abs_sum_variation)

    def __calculate_shuffled(self, shuffle_index, shuffle_df, average_correlation, abs_sum_correlation):
        time_correlation_df = pd.DataFrame(columns=list(range(self.seconds)), index=list(range(self.seconds)))

        shuffle_cell_correlations = []
        for second in range(self.start, self.end):
            frame_index = second * 10
            shuffle_corr = calculate_cell_correlation(shuffle_df, frame_index)

            shuffle_cell_correlations.append(shuffle_corr)

        calculate_time_correlation(self.seconds, time_correlation_df, shuffle_cell_correlations)

        average_correlation.iloc[shuffle_index, :] = time_correlation_df.mean()
        abs_sum_correlation.iloc[shuffle_index, :] = time_correlation_df.abs().sum()

    def __shuffle_default_df(self):
        return pd.DataFrame(columns=list(range(self.seconds)), index=list(range(self.SHUFFLE_COUNT)))
