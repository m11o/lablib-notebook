import pandas as pd
import numpy as np
import re
from os.path import basename, splitext
from math import isnan

import utils.sulfur.constant as const


class CaImageCSV:
    CELL_NAME_ROW_INDEX = 2
    ENGRAM_ROW_INDEX = 1

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.animal_number = self.__fetch_animal_number_from_path()
        self.data_frame = self.read_csv()
        self.engrams = self.__load_engrams()
        self.cell_names = list(self.engrams.keys())
        self.contexts = self.__load_contexts()

        self.__drop_useless_rows()

        self.index = self.__build_index()
        self.__drop_useless_columns()

        self.data_frame.set_index(self.index, inplace=True)
        self.data_frame.columns = self.cell_names

        self.__sort_data_frame()

        self.data_frame = self.data_frame.astype(float, copy=True)

    def read_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path, header=None, index_col=None, low_memory=False)

    def filtered_by_context(self, context_name) -> pd.DataFrame:
        if context_name not in self.contexts:
            raise ValueError('non-existence contexts')

        context_level = self.data_frame.index.get_loc_level(key=context_name, level=0, drop_level=False)[0]
        indexes = np.where(context_level)[0]

        return self.data_frame.iloc[indexes[0]:(indexes[-1] + 1), :]

    def is_engram(self, cell_name) -> bool:
        return self.engrams[cell_name]

    def engram_cells(self):
        return self.__fetch_engram_cell_names(is_engram=True)

    def non_engram_cells(self):
        return self.__fetch_engram_cell_names(is_engram=False)

    def __fetch_engram_cell_names(self, is_engram=True):
        return [cell_name for cell_name, value in self.engrams.items() if is_engram == value]

    def __fetch_animal_number_from_path(self):
        file_name = splitext(basename(self.csv_path))[0]
        matched = re.match('^(ID[0-9]{6}Cre[A-Z])_.+$', file_name)
        if not matched:
            raise ValueError('不正なANIMAL IDです')

        return matched.group(1)

    def __sort_data_frame(self):
        self.data_frame.sort_index(axis=0, level=[1, 0], inplace=True)
        self.data_frame.sort_index(axis=1, inplace=True)

    def __build_index(self) -> pd.MultiIndex:
        pre_index = self.data_frame.iloc[:, 1:3]
        pre_index.iloc[:, 1] = pre_index.iloc[:, 1].astype(float)
        context_series = pre_index.iloc[:, 0]
        for context_name in const.CONTEXTS:
            context_indexes = context_series[context_series == context_name].index
            start_index = context_indexes[0]
            end_index = context_indexes[-1]
            context_series.loc[start_index:end_index] = context_name

        return pd.MultiIndex.from_frame(pre_index, names=['context', 'time'])

    def __drop_useless_rows(self):
        self.data_frame.drop(index=[0, 1, 2], inplace=True)

    def __drop_useless_columns(self):
        self.data_frame.drop(columns=[0, 1, 2], inplace=True)

    def __load_engrams(self) -> dict:
        engrams = {}
        for _column_name, items in self.data_frame.iloc[1:3, :].iteritems():
            if type(items[self.CELL_NAME_ROW_INDEX]) is not str:
                continue

            matched = re.match('\s?(C[0-9]{3})', items[self.CELL_NAME_ROW_INDEX])
            if not matched:
                continue

            engrams[matched.group(1)] = not isnan(float(items[self.ENGRAM_ROW_INDEX]))

        sorted_list = sorted(engrams.items(), key=lambda x:x[0])
        return dict(sorted_list)

    def __load_contexts(self) -> list:
        contexts = self.data_frame.iloc[:, 1]
        contexts.dropna(inplace=True)
        self.__strip_list(contexts)

        return contexts.unique()

    @staticmethod
    def optimize_std(data_frame):
        for cell_name, items in data_frame.iteritems():
            if items.isnull().all() or (items == 0.0).all():
                continue
            data_frame.loc[:, cell_name] = CaImageCSV.optimize_std_by_series(items)

        return data_frame

    @staticmethod
    def optimize_std_by_series(series) -> pd.Series:
        series = series.astype(float)
        std = series.std(skipna=False)
        series /= std

        return series

    @staticmethod
    def __strip_list(array) -> None:
        for index, value in enumerate(array):
            array[index] = value.strip()

    def __repr__(self):
        return "<CaImageCSV csv_path=%s>" % self.csv_path
