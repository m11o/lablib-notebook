import pandas as pd
import numpy as np

from time_correlation_tool import calculate_time_correlation, build_cell_correlation_df


class ShuffleTimeCorrelationCalculator:
    SHUFFLE_COUNT = 1000

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
        shuffle_cell_correlations = build_cell_correlation_df(shuffle_df, self.start, self.end)

        time_correlation_df = pd.DataFrame(
            calculate_time_correlation(shuffle_cell_correlations),
            columns=list(range(self.seconds)),
            index=list(range(self.seconds))
        )
        self.shuffle_dfs.append(time_correlation_df)
