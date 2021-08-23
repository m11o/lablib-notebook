import pandas as pd
import numpy as np

import pystan as stan


class MatchingScoreResampling:
    MODEL_CODE = """
    data {
      int N;
      vector<lower=0>[N] Y;
    }
    
    parameters {
      real<lower=0> mu;
      real sigma;
    }
    
    model {
      Y ~ normal(mu, sigma);
    }
    """

    DEFAULT_ITERATION_COUNT = 1000
    DEFAULT_CHAINS = 3
    DEFAULT_WARMUP = 300

    def __init__(self):
        self.model = stan.StanModel(model_code=self.MODEL_CODE)

    def resamplings(self, matrix: pd.DataFrame):
        results = {}
        for context_name, series in matrix.iteritems():
            fit = self.resampling(series)
            mu = fit.extract('mu')['mu']
            results[context_name] = mu

        return results

    def resampling(self, series: pd.Series, iterator=DEFAULT_ITERATION_COUNT, chains=DEFAULT_CHAINS, warmup=DEFAULT_WARMUP):
        item_size = len(series)
        stan_data = {
            'N': item_size,
            'Y': series.values.tolist()
        }

        return self.model.sampling(data=stan_data, iter=iterator, chains=chains, warmup=warmup)

