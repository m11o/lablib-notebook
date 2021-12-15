import numpy as np
import pandas as pd

class XYDataParser:
    def __init__(self, filename) -> None:
        print('ok')
        self.filename = filename
        self.df = pd.read_csv(filename, header=[35])
