import pandas as pd


class MatrixOptimizer:
    def __init__(self, matrix):
        self.matrix = matrix.astype(float)

    def normalized(self) -> pd.DataFrame:
        def optimizer(series) -> pd.Series:
            std = series.std(skipna=False)
            mean = series.mean(skipna=False)
            series = (series - mean) / std

            return series

        return self.__optimized(optimizer)

    def divide_sd(self) -> pd.DataFrame:
        def optimizer(series) -> pd.Series:
            std = series.std(skipna=False)
            series /= std

            return series

        return self.__optimized(optimizer)

    def __optimized(self, series_optimizer) -> pd.DataFrame:
        def is_all_null_or_zero(series) -> bool:
            return series.isnull().all() or (series == 0.0).all()

        for cell_name, items in self.matrix.iteritems():
            items = items.astype(float)

            if is_all_null_or_zero(items):
                continue

            self.matrix.loc[:, cell_name] = series_optimizer(items)

        return self.matrix
