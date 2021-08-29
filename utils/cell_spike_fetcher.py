import numpy as np
import pandas as pd
from oasis.functions import deconvolve


class CellSpikeFetcher:
    def __init__(self, spike_threshold=0.05):
        self.fetched_spikes = {}
        self.restricted_fetched_spikes = {}
        self.spike_threshold = spike_threshold

    def fetch(self, cell_name, spike_series: pd.Series):
        if cell_name in self.fetched_spikes.keys():
            return self.fetched_spikes[cell_name]

        spikes = self.__estimate_spikes(spike_series)
        self.__append_preserved_shock_values(cell_name, spikes)

        return spikes

    def __estimate_spikes(self, spike_series: pd.Series):
        if spike_series.isnull().all():
            return np.zeros(len(spike_series))

        _c, spikes, _b, _g, _lam = deconvolve(spike_series.to_numpy(), g=(None, None), penalty=1)
        spikes[spikes <= self.spike_threshold] = 0.0

        return spikes

    def __append_preserved_shock_values(self, cell_name, spikes):
        if cell_name in self.fetched_spikes:
            return

        self.fetched_spikes[cell_name] = spikes
