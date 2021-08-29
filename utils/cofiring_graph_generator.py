import pandas as pd
import numpy as np
import networkx as nx

from cell_spike_fetcher import CellSpikeFetcher


class CofiringGraphGenerator:
    SPIKE_THRESHOLD = 0.05
    PEARSON_R_THRESHOLD = 0.1

    def __init__(self, matrix: pd.DataFrame, is_engram):
        self.matrix = matrix
        self.is_engram = is_engram

        self.spike_fetcher = CellSpikeFetcher(self.SPIKE_THRESHOLD)

    def run(self, start_frame: int, end_frame: int):
        graph = nx.Graph()
        estimated_spikes_df = self.__estimate_spikes(start_frame, end_frame)
        corr_df = self.__optimize_correlation_df(estimated_spikes_df.corr())

        cell_names = corr_df.columns
        for i in range(len(corr_df)):
            base_cell_name = cell_names[i]
            graph.add_node(base_cell_name, color=self.__cell_color_name(base_cell_name))

            for j in range(i + 1, len(corr_df)):
                weight = corr_df.iloc[i, j]
                if weight == 0.0 or np.isnan(weight):
                    continue

                compared_cell_name = cell_names[j]
                graph.add_edge(base_cell_name, compared_cell_name, weight=weight, length=1.0 / weight)

        return graph

    def __estimate_spikes(self, start_frame: int, end_frame: int):
        spikes_df = pd.DataFrame(index=list(range(start_frame, end_frame)), columns=self.matrix.columns)
        for cell_name, values in self.matrix.iteritems():
            spikes = self.spike_fetcher.fetch(cell_name, values)
            spikes_df.loc[:, cell_name] = spikes[start_frame:end_frame]

        return spikes_df

    def __optimize_correlation_df(self, corr_df):
        corr_df[corr_df < self.PEARSON_R_THRESHOLD] = 0.0
        corr_df.fillna(0.0, inplace=True)
        corr_df -= np.eye(len(corr_df))

        return corr_df

    def __cell_color_name(self, cell_name):
        if self.is_engram(cell_name):
            return 'firebrick'
        else:
            return 'royalblue'

