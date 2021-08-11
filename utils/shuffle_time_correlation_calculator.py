import pandas as pd
import numpy as np

from time_correlation_tool import calculate_time_correlation, calculate_cell_correlation


class ShuffleTimeCorrelationCalculator:
    SHUFFLE_COUNT = 100

    def __init__(self, all_df, engram_df, non_engram_df, start, end):
        self.all_df = all_df
        self.engram_df = engram_df
        self.non_engram_df = non_engram_df
        self.start = start
        self.end = end
        self.seconds = self.end - self.start

        self.average_variation_for_engram = self.__shuffle_default_df()
        self.abs_sum_variation_for_engram = self.__shuffle_default_df()

        self.average_variation_for_non_engram = self.__shuffle_default_df()
        self.abs_sum_variation_for_non_engram = self.__shuffle_default_df()

    def calc(self):
        for n in range(self.SHUFFLE_COUNT):
            random_engram_cells = np.random.randint(0, len(self.all_df.columns), size=len(self.engram_df.columns))
            shuffled_cell_df = self.all_df.iloc[:, random_engram_cells].copy(deep=True)

            self.__calculate_shuffled(n, shuffled_cell_df,
                                      self.average_variation_for_engram,
                                      self.abs_sum_variation_for_engram)

            random_non_engram_cells = np.random.randint(0, len(self.all_df.columns), size=len(self.non_engram_df.columns))
            shuffled_cell_df = self.all_df.iloc[:, random_non_engram_cells].copy(deep=True)

            self.__calculate_shuffled(n, shuffled_cell_df,
                                      self.average_variation_for_non_engram,
                                      self.abs_sum_variation_for_non_engram)

    def engram_quantile_for_95(self):
        average_quantile = pd.Series(index=self.average_variation_for_engram.columns)
        abs_sum_quantile = pd.Series(index=self.abs_sum_variation_for_engram.columns)

        for index, values in self.average_variation_for_engram.iteritems():
            average_quantile[index] = np.quantile(values, 0.95)

        for index, values in self.abs_sum_variation_for_engram.iteritems():
            abs_sum_quantile[index] = np.quantile(values, 0.95)

        return [average_quantile, abs_sum_quantile]

    def non_engram_quantile_for_95(self):
        average_quantile = pd.Series(index=self.average_variation_for_non_engram.columns)
        abs_sum_quantile = pd.Series(index=self.abs_sum_variation_for_non_engram.columns)

        for index, values in self.average_variation_for_non_engram.iteritems():
            average_quantile[index] = np.quantile(values, 0.95)

        for index, values in self.abs_sum_variation_for_non_engram.iteritems():
            abs_sum_quantile[index] = np.quantile(values, 0.95)

        return [average_quantile, abs_sum_quantile]

    def __calculate_quantiles(self, quantile_value, df, columns=None) -> pd.Series:
        if columns is None:
            columns = df.columns
        quantile_series = pd.Series(index=columns)

        for index, values in df.iteritems():
            quantile_series[index] = np.quantile(values, quantile_value)

        return quantile_series

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
