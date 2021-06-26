import pandas as pd
import numpy as np


class CalculateEventRate:
    EVENT_RATE_THRESHOLD = 1.5

    def __init__(self, animal_name, context_name, engrams=None, csv_path=None, threshold=0.0):
        self.animal_name = animal_name
        self.context_name = context_name
        self.engrams = engrams
        self.threshold = threshold
        self.csv_path = csv_path if csv_path is not None else self.build_csv_path()

        self.data_frame = pd.read_csv(self.csv_path, header=[0], index_col=[0, 1], low_memory=False)

        self.times = len(self.data_frame) / 10.0
        self.event_rates_by_cells = np.array([])
        self.engram_event_rates_by_cells = np.array([])
        self.non_engram_event_rates_by_cells = np.array([])

        self.event_rate = 0.0
        self.engram_event_rate = 0.0
        self.non_engram_event_rate = 0.0

        self.excluded_cells = []

    def calc_event_rate(self, excluded_cells=[]):
        for cell_name, values in self.data_frame.iteritems():
            values = values.astype(float)
            if values.isnull().all():
                self.excluded_cells.append(cell_name)
                continue

            if cell_name in excluded_cells:
                continue

            if (values == 0).all():
                self.__append_event_rate(cell_name, 0.0)

            event_count = len(values[values > self.threshold])
            event_rate = event_count / self.times

            # if event_rate >= self.EVENT_RATE_THRESHOLD:
            #     self.high_event_rate_cells.append(cell_name)

            self.__append_event_rate(cell_name, event_rate)

        self.event_rate, self.engram_event_rate, self.non_engram_event_rate = self.__mean_event_rate_by_cells()

    def build_csv_path(self):
        return './resources/spikes_data/%s/%s_%s.csv' % (self.animal_name, self.animal_name, self.context_name)

    def __append_event_rate(self, cell_name, event_rate):
        self.event_rates_by_cells = np.append(self.event_rates_by_cells, event_rate)
        if self.engrams is not None:
            if self.__is_engram(cell_name):
                self.engram_event_rates_by_cells = np.append(self.engram_event_rates_by_cells, event_rate)
            else:
                self.non_engram_event_rates_by_cells = np.append(self.non_engram_event_rates_by_cells, event_rate)

    def __is_engram(self, cell_name) -> bool:
        if self.engrams is None:
            raise ValueError

        return self.engrams[cell_name]

    def __mean_event_rate_by_cells(self):
        return [
            np.mean(self.event_rates_by_cells),
            np.mean(self.engram_event_rates_by_cells),
            np.mean(self.non_engram_event_rates_by_cells)
        ]
