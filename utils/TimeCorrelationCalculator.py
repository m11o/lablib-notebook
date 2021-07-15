import pandas as pd
import numpy as np

from context_data_csv import ContextDataCSV
from matrix_optimizer import MatrixOptimizer


class TimeCorrelationCalculator:
    SHUFFLE_COUNT = 1000

    def __init__(self, animal_name, context_name, data_frame=None, engram_cells=None, non_engram_cells=None):
        self.animal_name = animal_name
        self.context_name = context_name

        if data_frame is not None and engram_cells is not None and non_engram_cells is not None:
            self.data_frame = data_frame
            self.engram_cells = engram_cells
            self.non_engram_cells = non_engram_cells
        else:
            csv = ContextDataCSV(animal_name, context_name)
            self.data_frame = csv.data_frame
            self.engram_cells = csv.engram_cells()
            self.non_engram_cells = csv.non_engram_cells()

    def calc(self, start, end):
        seconds = end - start

        df = MatrixOptimizer(self.data_frame.copy(deep=True)).divide_sd()
        engram_df = self.__filtered_by_cells(df, self.engram_cells)
        non_engram_df = self.__filtered_by_cells(df, self.non_engram_cells)

        self.__dropna_and_fillna(df)
        self.__dropna_and_fillna(engram_df)
        self.__dropna_and_fillna(non_engram_df)

        corr_df_for_engram = pd.DataFrame(columns=list(range(seconds)), index=list(range(seconds)))
        corr_df_for_non_engram = pd.DataFrame(columns=list(range(seconds)), index=list(range(seconds)))

        engram_cells_correlations = []
        non_engram_cells_correlations = []
        for second in range(start, end):
            frame_index = second * 10
            engram_corr = self.__calculate_cell_correlation(engram_df, frame_index)
            non_engram_corr = self.__calculate_cell_correlation(non_engram_df, frame_index)

            engram_cells_correlations.append(engram_corr)
            non_engram_cells_correlations.append(non_engram_corr)

        self.__calc_time_correlation(seconds, corr_df_for_engram, engram_cells_correlations)
        self.__calc_time_correlation(seconds, corr_df_for_non_engram, non_engram_cells_correlations)

        corr_df_for_shuffle = pd.DataFrame(columns=list(range(seconds)), index=list(range(seconds)))
        for n in range(self.SHUFFLE_COUNT):
            random_cells = np.random.randint(0, len(df.columns), size=len(engram_df.columns))
            shuffle_df = df.iloc[:, random_cells].copy(deep=True)
            shuffle_cells_correlations = []
            for second in range(start, end):
                frame_index = second * 10
                shuffle_corr = self.__calculate_cell_correlation(shuffle_df, frame_index)

                shuffle_cells_correlations.append(shuffle_corr)

            self.__calc_time_correlation(seconds, corr_df_for_shuffle, shuffle_cells_correlations)

        corr_df_for_shuffled_engram = pd.DataFrame(columns=list(range(seconds)), index=list(range(seconds)))
        for n in range(self.SHUFFLE_COUNT):
            random_time = np.random.randint(0, len(engram_df.index), size=len(engram_df.index))
            shuffle_df = engram_df.iloc[random_time, :].copy(deep=True)
            shuffle_times_correlations = []
            for second in range(start, end):
                frame_index = second * 10
                shuffle_corr = self.__calculate_cell_correlation(shuffle_df, frame_index)

                shuffle_times_correlations.append(shuffle_corr)

            self.__calc_time_correlation(seconds, corr_df_for_shuffled_engram, shuffle_times_correlations)

        return [corr_df_for_shuffle / self.SHUFFLE_COUNT, corr_df_for_shuffled_engram / self.SHUFFLE_COUNT, corr_df_for_engram, corr_df_for_non_engram]

    @staticmethod
    def __calc_time_correlation(seconds, correlation_for_time, correlation_for_cells):
        for i in range(seconds):
            i_time_data = correlation_for_cells[i]
            for j in range(seconds):
                j_time_data = correlation_for_cells[j]

                if correlation_for_time.at[j, i] is not np.nan:
                    correlation_for_time.at[i, j] = correlation_for_time.at[j, i]
                    continue

                cell_count = len(i_time_data)
                calculated_values = i_time_data * j_time_data
                corr_sum = (calculated_values - (calculated_values * np.eye(len(calculated_values)))).sum().sum()
                if correlation_for_time.at[i, j] is np.nan:
                    correlation_for_time.at[i, j] = 0.0
                correlation_for_time.at[i, j] += corr_sum / (cell_count * (cell_count - 1))

    @staticmethod
    def __calculate_correlation_for_cells(df, engram_df, non_engram_df, frame_index, size=9):
        all_sliding_window = df.iloc[frame_index:frame_index+size, :]
        engram_sliding_window = engram_df.iloc[frame_index:frame_index+size, :]
        non_engram_sliding_window = non_engram_df.iloc[frame_index:frame_index+size, :]

        return [all_sliding_window.corr(), engram_sliding_window.corr(), non_engram_sliding_window.corr()]

    @staticmethod
    def __calculate_cell_correlation(df, frame_index, size=9):
        sliding_window = df.iloc[frame_index:frame_index+size, :]
        return sliding_window.corr()

    @staticmethod
    def __filtered_by_cells(df, cell_names) -> pd.DataFrame:
        return df.loc[:, cell_names]

    @staticmethod
    def __dropna_and_fillna(matrix):
        matrix.dropna(how='all', axis=1, inplace=True)
        matrix.fillna(0.0, inplace=True, axis=1)

