import pandas as pd


class ContextDataCSV:
    def __init__(self, animal_name, context_name, csv_file_path=None):
        self.animal_name = animal_name
        self.context_name = context_name
        self.csv_file_path = self.__build_csv_path() if csv_file_path is None else csv_file_path

        self.data_frame = pd.read_csv(self.csv_file_path, header=None, index_col=None, low_memory=False)
        self.engrams = self.__load_engrams()
        self.cell_names = list(self.engrams.keys())

        self.__drop_useless_rows()
        self.__drop_useless_columns()
        self.data_frame.reset_index(inplace=True, drop=True)
        self.data_frame.columns = self.cell_names

        self.data_frame = self.data_frame.astype(float, copy=True)

    def is_engram(self, cell_name) -> bool:
        return self.engrams[cell_name]

    def engram_cells(self):
        return self.__fetch_engram_cell_names(is_engram=True)

    def non_engram_cells(self):
        return self.__fetch_engram_cell_names(is_engram=False)

    def __fetch_engram_cell_names(self, is_engram=True):
        return [cell_name for cell_name, value in self.engrams.items() if is_engram == value]

    def __build_csv_path(self):
        return './resources/context_data/%s/%s.csv' % (self.animal_name, self.context_name)

    def __load_engrams(self) -> dict:
        engrams = {}
        for _, items in self.data_frame.iloc[:2, 2:].iteritems():
            engrams[items[0]] = items[1] == 'True'

        sorted_list = sorted(engrams.items(), key=lambda x: x[0])
        return dict(sorted_list)

    def __drop_useless_rows(self):
        self.data_frame.drop(index=[0, 1, 2], inplace=True)

    def __drop_useless_columns(self):
        self.data_frame.drop(columns=[0, 1], inplace=True)

    def __build_index(self) -> pd.MultiIndex:
        pre_index = self.data_frame.iloc[:, 0:2]
        pre_index.iloc[:, 1] = pre_index.iloc[:, 1].astype(float)

        return pd.MultiIndex.from_frame(pre_index, names=['context', 'time'])
