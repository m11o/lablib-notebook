import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns


class TimeCorrelationViewer:
    SCALE_MIN = -1
    SCALE_MAX = 1
    HEATMAP_CMAP = 'OrRd'

    def __init__(self, shuffle_df, engram_df, non_engram_df):
        self.shuffle_df = shuffle_df.astype(float)
        self.engram_df = engram_df.astype(float)
        self.non_engram_df = non_engram_df.astype(float)

        self.max_value, self.min_value = self.__calc_minmax()
        self.scaled_shuffle_df, self.scaled_engram_df, self.scaled_non_engram_df = self.scale_df(shuffle_df, engram_df, non_engram_df)

    def view_heatmap(self):
        _fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(25, 7))
        ax_for_all = sns.heatmap(self.scaled_shuffle_df, ax=axes[0], robust=True, vmin=self.min_value, vmax=self.max_value, square=True, cmap=self.HEATMAP_CMAP)
        ax_for_all.invert_yaxis()
        ax_for_engram = sns.heatmap(self.scaled_engram_df, ax=axes[1], robust=True, vmin=self.min_value, vmax=self.max_value, square=True, cmap=self.HEATMAP_CMAP)
        ax_for_engram.invert_yaxis()
        ax_for_non_engram = sns.heatmap(self.scaled_non_engram_df, ax=axes[2], robust=True, vmin=self.min_value, vmax=self.max_value, square=True, cmap=self.HEATMAP_CMAP)
        ax_for_non_engram.invert_yaxis()
        plt.show()

    def view_correlation_colormesh(self):
        fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(25, 7))

        self.__view_colormesh(self.shuffle_df, axes[0], 'all cells')
        self.__view_colormesh(self.engram_df, axes[1], 'engram cells')
        a = self.__view_colormesh(self.non_engram_df, axes[2], 'non-engram cells')
        plt.colorbar(a, orientation="vertical")

        plt.show()

    def view_sum_of_correlation_bar(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 15))

        shuffle_sum = self.shuffle_df.sum()
        engram_sum = self.engram_df.sum()
        non_engram_sum = self.non_engram_df.sum()

        max_value = max([shuffle_sum.max(), engram_sum.max(), non_engram_sum.max()])
        axes[0][0].plot(shuffle_sum, label='all cells')
        axes[0][0].plot(engram_sum, label='engram cells')
        axes[0][0].plot(non_engram_sum, label='non-engram cells')
        axes[0][0].set_ylim(0, max_value + 1)
        axes[0][0].legend()

        shuffle_mean = self.shuffle_df.mean()
        engram_mean = self.engram_df.mean()
        non_engram_mean = self.non_engram_df.mean()

        axes[0][1].plot(shuffle_mean, label='all cells')
        axes[0][1].plot(engram_mean, label='engram cells')
        axes[0][1].plot(non_engram_mean, label='non-engram cells')
        axes[0][1].legend()

        sns.distplot(engram_sum, bins=5, label='engram cells', ax=axes[1][0])
        sns.distplot(non_engram_sum, bins=5, label='non-engram cells', ax=axes[1][0])
        axes[1][0].legend()

        engram_sum_df = pd.DataFrame(engram_sum).assign(Cell='engram cells')
        non_engram_sum_df = pd.DataFrame(non_engram_sum).assign(Cell='non-engram cells')
        concat_df = pd.concat([engram_sum_df, non_engram_sum_df])
        data = pd.melt(concat_df, id_vars=['Cell'], var_name=['Number'], value_name='sum of correlation')
        sns.violinplot(x='Cell', y='sum of correlation', data=data, ax=axes[1][1])
        sns.swarmplot(x='Cell', y='sum of correlation', data=data, ax=axes[1][1], color='0.5')
        fig.show()

    def __calc_colormesh_maxmin(self):
        all_average = self.calc_diagonal_average(self.shuffle_df)
        engram_average = self.calc_diagonal_average(self.engram_df)
        non_engram_average = self.calc_diagonal_average(self.non_engram_df)

        return np.array([all_average, engram_average, non_engram_average]).mean()

    def __calc_minmax(self):
        all_max = self.shuffle_df.max().max()
        engram_max = self.engram_df.max().max()
        non_engram_max = self.non_engram_df.max().max()

        max_value = max(all_max, engram_max, non_engram_max)

        all_min = self.shuffle_df.min().min()
        engram_min = self.engram_df.min().min()
        non_engram_min = self.non_engram_df.min().min()

        min_value = min(all_min, engram_min, non_engram_min)

        return max_value, min_value

    @staticmethod
    def calc_diagonal_average(df):
        seconds = len(df)
        return (df * np.eye(seconds)).sum().mean()

    @staticmethod
    def scale_df(shuffle_df, engram_df, non_engram_df):
        seconds = len(non_engram_df)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(non_engram_df)
        scaled_shuffle_df = scaler.transform(shuffle_df)
        scaled_shuffle_df = pd.DataFrame(scaled_shuffle_df.astype(float), columns=list(range(seconds)), index=list(range(seconds)))
        scaled_engram_df = scaler.transform(engram_df)
        scaled_engram_df = pd.DataFrame(scaled_engram_df.astype(float), columns=list(range(seconds)), index=list(range(seconds)))
        scaled_non_engram_df = scaler.transform(non_engram_df)
        scaled_non_engram_df = pd.DataFrame(scaled_non_engram_df.astype(float), columns=list(range(seconds)), index=list(range(seconds)))

        return [scaled_shuffle_df, scaled_engram_df, scaled_non_engram_df]

    @staticmethod
    def __view_colormesh(df, axe, title=''):
        x = y = np.arange(0, len(df))
        X, Y = np.meshgrid(x, y)

        df = df.astype(float)
        axe.set_title(title)
        return axe.pcolormesh(X, Y, df, shading='auto', cmap='bwr', vmin=-1.0, vmax=1.0)
