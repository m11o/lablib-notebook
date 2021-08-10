import pandas as pd
import numpy as np

import utils.sulfur.constant as const

from utils.context_data_csv import ContextDataCSV
from utils.matrix_optimizer import MatrixOptimizer

from utils.useful_tool import find_all_nan_cells, dropped_unique_cells, calc_event_rate

from OASIS.oasis.functions import deconvolve
from scipy.stats import pearsonr


class EstimateShockCell:
    SHOCK_CELL_RATE_COLUMNS = ['A1_1', 'A1_2', 'A1_3', 'A1_all', 'A4_1', 'A4_2', 'A4_3', 'A4_all', 'across_all']
    SHUFFLED_COUNT = 1000
    SPIKE_THRESHOLD = 0.05
    COFIRING_THRESHOLD = 0.0
    STD_PRODUCT = 2.0

    FIRST_SHOCK_START_FRAME = 80
    FIRST_SHOCK_END_FRAME = 100

    SECOND_SHOCK_START_FRAME = 980
    SECOND_SHOCK_END_FRAME = 1000

    THIRD_SHOCK_START_FRAME = 1880
    THIRD_SHOCK_END_FRAME = 1900

    def __init__(self, animal_name, cells, data_frame):
        self.animal_name = animal_name
        self.cells = cells
        self.shock_cell_rate_df = data_frame

    def run(self):
        a1_matrix, a4_matrix = self.__init_context_matrix()
        a1_cofiring_cells_all = self.__find_cofiring_shock_cells(a1_matrix, 'A1')
        a4_cofiring_cells_all = self.__find_cofiring_shock_cells(a4_matrix, 'A4')

        print(a1_cofiring_cells_all)
        print(a4_cofiring_cells_all)
        across_cells = []
        for cell_name in a1_cofiring_cells_all:
            if cell_name in a4_cofiring_cells_all:
                across_cells.append(cell_name)

        print(across_cells)
        self.__append_shock_cell_rate('across_all', across_cells)

    def __init_context_matrix(self):
        a1_csv = ContextDataCSV(self.animal_name, 'A1postES')
        a1_matrix = MatrixOptimizer(a1_csv.data_frame).divide_sd()
        a1_matrix = a1_matrix.loc[:, self.cells]

        a4_csv = ContextDataCSV(self.animal_name, 'A4postES')
        a4_matrix = MatrixOptimizer(a4_csv.data_frame).divide_sd()
        a4_matrix = a4_matrix.loc[:, self.cells]

        a1_nan_cells = find_all_nan_cells(a1_matrix)
        a4_nan_cells = find_all_nan_cells(a4_matrix)
        dropped_cells = dropped_unique_cells(a1_nan_cells, a4_nan_cells)
        a1_matrix.drop(columns=dropped_cells, inplace=True)
        a4_matrix.drop(columns=dropped_cells, inplace=True)

        return a1_matrix, a4_matrix

    def __find_cofiring_shock_cells(self, matrix, context_name):
        preserved_spikes = {}

        upper_event_rate_cells_1 = []
        upper_event_rate_cells_2 = []
        upper_event_rate_cells_3 = []

        cofiring_cells_all = []

        for cell_name, values in matrix.iteritems():
            if self.__is_all_nan_or_zero(values):
                continue

            spikes = self.__fetch_estimated_spikes(cell_name, preserved_spikes, values)

            shock_event_rate_1 = self.__shock_event_rate(spikes, 0)
            shock_event_rate_2 = self.__shock_event_rate(spikes, 1)
            shock_event_rate_3 = self.__shock_event_rate(spikes, 2)

            shuffled_event_rate_mean, shuffled_event_rate_std = self.__shuffled_event_rate(values)

            event_rate_threshold = shuffled_event_rate_mean + (shuffled_event_rate_std * self.STD_PRODUCT)
            if event_rate_threshold <= shock_event_rate_1:
                upper_event_rate_cells_1.append(cell_name)
            if event_rate_threshold <= shock_event_rate_2:
                upper_event_rate_cells_2.append(cell_name)
            if event_rate_threshold <= shock_event_rate_3:
                upper_event_rate_cells_3.append(cell_name)

        cofiring_cells_1 = self.__estimate_cofiring_shock_cells(upper_event_rate_cells_1, matrix, preserved_spikes, timing_level=0)
        cofiring_cells_2 = self.__estimate_cofiring_shock_cells(upper_event_rate_cells_2, matrix, preserved_spikes, timing_level=1)
        cofiring_cells_3 = self.__estimate_cofiring_shock_cells(upper_event_rate_cells_3, matrix, preserved_spikes, timing_level=2)

        for cell_name in cofiring_cells_1:
            if cell_name in cofiring_cells_2 and cell_name in cofiring_cells_3:
                cofiring_cells_all.append(cell_name)

        self.__append_shock_cell_rate('%s_1' % context_name, cofiring_cells_1)
        self.__append_shock_cell_rate('%s_2' % context_name, cofiring_cells_2)
        self.__append_shock_cell_rate('%s_3' % context_name, cofiring_cells_3)
        self.__append_shock_cell_rate('%s_all' % context_name, cofiring_cells_all)

        return cofiring_cells_all

    def __append_shock_cell_rate(self, column_name, cofiring_cells):
        self.shock_cell_rate_df.loc[self.animal_name, column_name] = len(cofiring_cells) / len(self.cells)

    def __fetch_estimated_spikes(self, cell_name, preserved_spikes, values):
        spikes = self.__fetch_preserved_spikes(cell_name, preserved_spikes)
        if spikes is None:
            spikes = self.__estimate_spikes(values)

        self.__append_preserved_spikes(cell_name, spikes, preserved_spikes)
        return spikes

    def __estimate_spikes(self, values):
        if self.__is_all_nan_or_zero(values):
            return np.zeros(len(values))

        _c, spikes, _b, _g, _lam = deconvolve(values.to_numpy(), g=(None, None), penalty=1)
        spikes[spikes <= self.SPIKE_THRESHOLD] = 0.0

        return spikes

    def __estimate_cofiring_shock_cells(self, cells, matrix, preserved_spikes, timing_level=0):
        start_frame, end_frame = self.__shock_timing(timing_level)

        cofiring_cells = []
        for base_shock_cell in cells:
            base_values = matrix.loc[:, base_shock_cell]
            base_shock_spikes = self.__fetch_estimated_spikes(base_shock_cell, preserved_spikes, base_values)
            base_shock_spikes = base_shock_spikes[start_frame:end_frame]

            for compared_shock_cell in cells:
                if base_shock_cell == compared_shock_cell:
                    continue

                compared_values = matrix.loc[:, compared_shock_cell]
                compared_shock_spikes = self.__fetch_estimated_spikes(compared_shock_cell, preserved_spikes, compared_values)
                compared_shock_spikes = compared_shock_spikes[start_frame:end_frame]

                r, _ = pearsonr(base_shock_spikes, compared_shock_spikes)
                if r > self.COFIRING_THRESHOLD:
                    cofiring_cells.append(base_shock_cell)
                    break

        return cofiring_cells

    def __fetch_spikes_in_shock(self, spikes, timing_level=0):
        start_frame, end_frame = self.__shock_timing(timing_level)

        return spikes[start_frame:end_frame]

    def __shock_timing(self, timing_level=0):
        if timing_level == 0:
            return self.FIRST_SHOCK_START_FRAME, self.FIRST_SHOCK_END_FRAME
        elif timing_level == 1:
            return self.SECOND_SHOCK_START_FRAME, self.SECOND_SHOCK_END_FRAME
        elif timing_level == 2:
            return self.THIRD_SHOCK_START_FRAME, self.THIRD_SHOCK_END_FRAME

    def __shock_event_rate(self, spikes, timing_level=0):
        shock_spikes = self.__fetch_spikes_in_shock(spikes, timing_level)
        return calc_event_rate(shock_spikes, second=2.0)

    def __shuffled_event_rate(self, values_in_cell):
        shuffled_event_rates = np.array([])
        for _ in range(self.SHUFFLED_COUNT):
            random_time = np.random.randint(0, len(values_in_cell), size=len(values_in_cell))
            shuffled_items = values_in_cell[random_time]

            shuffled_spikes = self.__estimate_spikes(shuffled_items)
            shuffled_event_rate_1 = self.__shock_event_rate(shuffled_spikes, 0)
            shuffled_event_rate_2 = self.__shock_event_rate(shuffled_spikes, 1)
            shuffled_event_rate_3 = self.__shock_event_rate(shuffled_spikes, 2)

            shuffled_event_rate_summary = np.array([shuffled_event_rate_1, shuffled_event_rate_2, shuffled_event_rate_3])
            shuffled_event_rates = np.append(shuffled_event_rates, np.mean(shuffled_event_rate_summary))

        return np.mean(shuffled_event_rates), np.std(shuffled_event_rates)


    @staticmethod
    def __fetch_preserved_spikes(shock_cell_name, preserved_spikes):
        if shock_cell_name in preserved_spikes:
            return preserved_spikes[shock_cell_name]

        return None

    @staticmethod
    def __append_preserved_spikes(shock_cell_name, spikes, preserved_spikes):
        if shock_cell_name in preserved_spikes:
            return

        preserved_spikes[shock_cell_name] = spikes

    @staticmethod
    def __is_all_nan_or_zero(values):
        return values.isnull().all() or values[values == 0].all()

    @classmethod
    def init_shock_cell_rate_df(cls):
        return pd.DataFrame(index=const.ANIMAL_NAMES, columns=cls.SHOCK_CELL_RATE_COLUMNS)


