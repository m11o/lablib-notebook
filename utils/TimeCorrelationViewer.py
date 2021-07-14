import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns


class TimeCorrelationViewer:
    SCALE_MIN = -0.06
    SCALE_MAX = 0.06
    HEATMAP_CMAP = 'RdYlBu_r'

    DEFAULT_FONTSIZE = 20
    DEFAULT_FONT_WEIGHT = 'bold'

    def __init__(self, shuffle_df, engram_df, non_engram_df):
        self.shuffle_df = shuffle_df.astype(float)
        self.engram_df = engram_df.astype(float)
        self.non_engram_df = non_engram_df.astype(float)
        self.time_ticks = list(range(0, len(self.shuffle_df) + 1, 5))

    def view_heatmap(self):
        fig, axes = plt.subplots(ncols=3, figsize=(25, 7), sharey=True, tight_layout=False, dpi=200)

        ax_for_engram = sns.heatmap(self.engram_df, ax=axes[0], robust=True, vmin=self.SCALE_MIN, vmax=self.SCALE_MAX, square=True, cmap=self.HEATMAP_CMAP)
        axes[0].set_xticks(self.time_ticks)
        axes[0].set_xticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[0].set_xlabel('Reference Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[0].set_yticks(self.time_ticks)
        axes[0].set_yticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[0].set_ylabel('Compared Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[0].set_title('Engram cells', fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)

        color_bar_for_engram = ax_for_engram.collections[0].colorbar
        color_bar_for_engram.ax.tick_params(labelsize=self.DEFAULT_FONTSIZE)
        ax_for_engram.invert_yaxis()

        ax_for_non_engram = sns.heatmap(self.non_engram_df, ax=axes[1], robust=True, vmin=self.SCALE_MIN, vmax=self.SCALE_MAX, square=True, cmap=self.HEATMAP_CMAP)
        axes[1].set_xticks(self.time_ticks)
        axes[1].set_xticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[1].set_xlabel('Reference Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[1].set_yticks(self.time_ticks)
        axes[1].set_yticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[1].set_title('Non-engram cells', fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)

        color_bar_for_non_engram = ax_for_non_engram.collections[0].colorbar
        color_bar_for_non_engram.ax.tick_params(labelsize=self.DEFAULT_FONTSIZE)
        ax_for_non_engram.invert_yaxis()

        ax_for_shuffle = sns.heatmap(self.shuffle_df, ax=axes[2], robust=True, vmin=self.SCALE_MIN, vmax=self.SCALE_MAX,
                                     square=True, cmap=self.HEATMAP_CMAP)
        axes[2].set_xticks(self.time_ticks)
        axes[2].set_xticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[2].set_xlabel('Reference Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[2].set_yticks(self.time_ticks)
        axes[2].set_yticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[2].set_title('Shuffled cells', fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)

        color_bar_for_shuffle = ax_for_shuffle.collections[0].colorbar
        color_bar_for_shuffle.ax.tick_params(labelsize=self.DEFAULT_FONTSIZE)
        ax_for_shuffle.invert_yaxis()

        fig.show()

    def view_sum_of_correlation_bar(self):
        fig, ax = plt.subplots(1, figsize=(5, 5))

        abs_shuffle_sum = self.shuffle_df.abs().sum()
        abs_engram_sum = self.engram_df.abs().sum()
        abs_non_engram_sum = self.non_engram_df.abs().sum()

        ax.plot(abs_shuffle_sum, label='shuffled cells')
        ax.plot(abs_engram_sum, label='engram cells')
        ax.plot(abs_non_engram_sum, label='non-engram cells')
        ax.legend(loc='lower right')

        ax.set_title('Variation from shuffled cells', fontsize=16)
        ax.set_ylabel('Sum of correlations', fontsize=14)
        ax.set_xlabel('Reference time (s)', fontsize=14)
        fig.show()
