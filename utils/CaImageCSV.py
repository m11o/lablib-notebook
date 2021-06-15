import pandas as pd


class CaImageCSV:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data_frame = self.read_csv()

        self.cell_names = self.__load_cell_names()
        self.contexts = self.__load_contexts()

    def read_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path, header=[1, 2], index_col=[0, 1, 2], low_memory=False)

    def __load_cell_names(self):
        cell_names = self.data_frame.columns.levels[1].values
        self.__strip_list(cell_names)

        return cell_names

    def __load_contexts(self):
        contexts = self.data_frame.index.levels[1].values
        self.__strip_list(contexts)

        return contexts

    @staticmethod
    def __strip_list(array) -> None:
        for index, cell_name in enumerate(array):
            array[index] = cell_name.strip()

    def __repr__(self):
        return "<CaImageCSV csv_path=%s>" % self.csv_path
