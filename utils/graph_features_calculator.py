import numpy as np
import pandas as pd
import networkx as nx
from statistics import mean


class GraphFeaturesCalculator:
    WEIGHT_KEY = 'weight'
    LENGTH_KEY = 'length'

    def __init__(self, graph):
        self.graph = graph

    def calc_all_features(self):
        return {
            'clustering_coefficient': self.calc_clustering_coefficient(),
            'shortest_path_length': self.calc_shortest_path_length(),
            'cofiring_strength': self.calc_cofiring_strength(),
            'assortativity': self.calc_assortativity(),
            'total_weight': self.calc_total_weight(),
            'total_number_of_edges': self.calc_total_number_of_edges()
        }

    def calc_clustering_coefficient(self, weight_key=WEIGHT_KEY):
        return nx.average_clustering(self.graph, weight=weight_key)

    def calc_shortest_path_length(self, length_key=LENGTH_KEY):
        all_lengths = []
        for node in self.graph:
            for target in self.graph:
                if not nx.has_path(self.graph, node, target) or node == target:
                    continue

                all_lengths.append(nx.dijkstra_path_length(self.graph, node, target, weight=length_key))

        return mean(all_lengths)

    def calc_cofiring_strength(self):
        adjacency_matrix = pd.DataFrame(nx.adjacency_matrix(self.graph).todense())
        return adjacency_matrix.sum().mean()

    def calc_assortativity(self, weight_key=WEIGHT_KEY):
        return nx.degree_pearson_correlation_coefficient(self.graph, weight=weight_key)

    def calc_total_weight(self):
        return sum(weight for _, _, weight in self.graph.edges.data("weight"))

    def calc_total_number_of_edges(self):
        return len(self.graph.edges)
