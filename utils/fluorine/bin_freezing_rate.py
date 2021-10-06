import pandas as pd


class BinFreezingRate:
    def __init__(self):
        self.df = pd.DataFrame()
        self.header = []
        self.bin_number = 0

    def add_header(self, header_list):
        self.header = header_list
        self.bin_number = len(header_list[2:])

        self.df.columns = header_list

    def add_row(self, row):
        try:
            self.df.append(row, verify_integrity=True)
        except ValueError:
            print('already row in data_frame')
