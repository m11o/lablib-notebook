import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

import seaborn as sns


class TimeCorrelationViewer:
    SCALE_MIN = -0.06
    SCALE_MAX = 0.06
    HEATMAP_CMAP = 'RdYlBu_r'

    DEFAULT_FONTSIZE = 20
    DEFAULT_FONT_WEIGHT = 'bold'

    def __init__(self, shuffle_df, shuffled_engram_df, engram_df, non_engram_df, ticks_span=5):
        self.shuffle_df = shuffle_df.astype(float)
        self.shuffled_engram_df = shuffled_engram_df.astype(float)
        self.engram_df = engram_df.astype(float)
        self.non_engram_df = non_engram_df.astype(float)
        self.time_ticks = list(range(0, len(self.shuffle_df) + 1, ticks_span))

    def draw(self, title=None):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 16), dpi=200)

        ax_for_engram = sns.heatmap(self.engram_df, ax=axes[0][0], robust=True, vmin=self.SCALE_MIN, vmax=self.SCALE_MAX, square=True, cmap=self.HEATMAP_CMAP)
        axes[0][0].set_xticks(self.time_ticks)
        axes[0][0].set_xticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[0][0].set_xlabel('Reference Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[0][0].set_yticks(self.time_ticks)
        axes[0][0].set_yticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[0][0].set_ylabel('Compared Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[0][0].set_title('Engram cells', fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)

        color_bar_for_engram = ax_for_engram.collections[0].colorbar
        color_bar_for_engram.ax.tick_params(labelsize=15)
        ax_for_engram.invert_yaxis()

        ax_for_non_engram = sns.heatmap(self.non_engram_df, ax=axes[0][1], robust=True, vmin=self.SCALE_MIN, vmax=self.SCALE_MAX, square=True, cmap=self.HEATMAP_CMAP)
        axes[0][1].set_xticks(self.time_ticks)
        axes[0][1].set_xticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[0][1].set_xlabel('Reference Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[0][1].set_yticks(self.time_ticks)
        axes[0][1].set_yticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[0][1].set_title('Non-engram cells', fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)

        color_bar_for_non_engram = ax_for_non_engram.collections[0].colorbar
        color_bar_for_non_engram.ax.tick_params(labelsize=15)
        ax_for_non_engram.invert_yaxis()

        shuffle_mean = self.shuffle_df.mean()
        engram_mean = self.engram_df.mean()
        non_engram_mean = self.non_engram_df.mean()
        self.view_time_correlation_plot(axes[0][2], shuffle_mean, engram_mean, non_engram_mean, title='Average of time correlation')

        ax_for_shuffle_engram = sns.heatmap(self.shuffled_engram_df, ax=axes[1][0], robust=True, vmin=self.SCALE_MIN, vmax=self.SCALE_MAX, square=True, cmap=self.HEATMAP_CMAP)
        axes[1][0].set_xticks(self.time_ticks)
        axes[1][0].set_xticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[1][0].set_xlabel('Reference Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[1][0].set_yticks(self.time_ticks)
        axes[1][0].set_yticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[1][0].set_title('Shuffled engram cells', fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)

        color_bar_for_shuffle_engram = ax_for_shuffle_engram.collections[0].colorbar
        color_bar_for_shuffle_engram.ax.tick_params(labelsize=15)
        ax_for_shuffle_engram.invert_yaxis()

        ax_for_shuffle = sns.heatmap(self.shuffle_df, ax=axes[1][1], robust=True, vmin=self.SCALE_MIN, vmax=self.SCALE_MAX, square=True, cmap=self.HEATMAP_CMAP)
        axes[1][1].set_xticks(self.time_ticks)
        axes[1][1].set_xticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[1][1].set_xlabel('Reference Time (s)', fontsize=self.DEFAULT_FONTSIZE)
        axes[1][1].set_yticks(self.time_ticks)
        axes[1][1].set_yticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        axes[1][1].set_title('Shuffled cells', fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)

        color_bar_for_shuffle = ax_for_shuffle.collections[0].colorbar
        color_bar_for_shuffle.ax.tick_params(labelsize=15)
        ax_for_shuffle.invert_yaxis()

        abs_shuffle_sum = self.shuffle_df.abs().sum()
        abs_engram_sum = self.engram_df.abs().sum()
        abs_non_engram_sum = self.non_engram_df.abs().sum()
        self.view_time_correlation_plot(axes[1][2], abs_shuffle_sum, abs_engram_sum, abs_non_engram_sum, title='Variation from shuffle cells')

        if title is not None:
            fig.suptitle(title, fontsize=22, fontweight=self.DEFAULT_FONT_WEIGHT)
        fig.show()

    def view_time_correlation_plot(self, ax, shuffle_data, engram_data, non_engram_data, title):
        _, pvalue = ks_2samp(engram_data, non_engram_data)
        ax.text(0.05, 0.1, 'p = %f' % pvalue, transform=ax.transAxes, fontstyle='italic', fontweight='bold', fontsize=16)

        ax.plot(shuffle_data, label='shuffled cells')
        ax.plot(engram_data, label='engram cells')
        ax.plot(non_engram_data, label='non-engram cells')
        ax.legend(loc='lower right', fontsize=15)

        ax.set_title(title, fontsize=self.DEFAULT_FONTSIZE, fontweight=self.DEFAULT_FONT_WEIGHT)
        ax.set_xticks(self.time_ticks)
        ax.set_xticklabels(self.time_ticks, fontsize=self.DEFAULT_FONTSIZE)
        ax.set_xlabel('Reference Time (s)', fontsize=self.DEFAULT_FONTSIZE)

        ax.tick_params(axis='y', labelsize=self.DEFAULT_FONTSIZE)
