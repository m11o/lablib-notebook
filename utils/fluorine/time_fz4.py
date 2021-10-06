import pandas as pd

from bin_freezing_rate import BinFreezingRate
from time_fz4_matcher import is_freezing_rate_header, is_freezing_rate_row

class TimeFZ4:
    def __init__(self):
        self.user_name = ''
        self.experiment_date = None
        self.project_name = ''
        self.session_name = ''

        self.bin_freezing_rate = BinFreezingRate()

    @classmethod
    def read_file(cls, file_path):
        time_fz4 = cls()
        with open(file_path) as f:
            for line in f.readlines():
                if is_freezing_rate_header(line):
                    header = line.split('\t')
                    time_fz4.bin_freezing_rate.add_header(header)
                elif is_freezing_rate_row(line):
                    row = line.split('\t')
                    time_fz4.bin_freezing_rate.add_row(row)

        return time_fz4
