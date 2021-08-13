import pandas as pd
import numpy as np

from time_correlation_tool import calculate_time_correlation, calculate_cell_correlation


class ShuffleTimeCorrelationCalculator:
    SHUFFLE_COUNT = 10000

    def __init__(self, all_df, shuffle_cell_size, start, end):
        self.all_df = all_df
        self.shuffle_cell_size = shuffle_cell_size
        self.start = start
        self.end = end
        self.seconds = self.end - self.start

        self.shuffle_dfs = []

    def calc(self):
        for _ in range(self.SHUFFLE_COUNT):
            random_engram_cells = np.random.randint(0, len(self.all_df.columns), size=self.shuffle_cell_size)
            shuffled_cell_df = self.all_df.iloc[:, random_engram_cells].copy(deep=True)

            self.__calculate_shuffled(shuffled_cell_df)

    def __calculate_shuffled(self, shuffle_df):
        time_correlation_df = pd.DataFrame(columns=list(range(self.seconds)), index=list(range(self.seconds)))

        shuffle_cell_correlations = []
        for second in range(self.start, self.end):
            frame_index = second * 10
            shuffle_corr = calculate_cell_correlation(shuffle_df, frame_index)

            shuffle_cell_correlations.append(shuffle_corr)

        calculate_time_correlation(self.seconds, time_correlation_df, shuffle_cell_correlations)
        self.shuffle_dfs.append(time_correlation_df)
