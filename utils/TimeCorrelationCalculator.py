import pandas as pd

from context_data_csv import ContextDataCSV
from matrix_optimizer import MatrixOptimizer
from time_correlation_tool import calculate_cell_correlation, calculate_time_correlation
from shuffle_time_correlation_calculator import ShuffleTimeCorrelationCalculator


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
            engram_corr = calculate_cell_correlation(engram_df, frame_index)
            non_engram_corr = calculate_cell_correlation(non_engram_df, frame_index)

            engram_cells_correlations.append(engram_corr)
            non_engram_cells_correlations.append(non_engram_corr)

        calculate_time_correlation(seconds, corr_df_for_engram, engram_cells_correlations)
        calculate_time_correlation(seconds, corr_df_for_non_engram, non_engram_cells_correlations)

        shuffle_calculator = ShuffleTimeCorrelationCalculator(df, len(engram_df.columns), start, end)
        shuffle_calculator.calc()

        return [shuffle_calculator, corr_df_for_engram, corr_df_for_non_engram]

    @staticmethod
    def __filtered_by_cells(df, cell_names) -> pd.DataFrame:
        return df.loc[:, cell_names]

    @staticmethod
    def __dropna_and_fillna(matrix):
        matrix.dropna(how='all', axis=1, inplace=True)
        matrix.fillna(0.0, inplace=True, axis=1)

