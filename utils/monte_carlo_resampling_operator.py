import numpy as np
import pandas as pd

import pystan as stan


class MonteCarloResamplingOperator:
    MODEL_CODE = """
    data {
      int N;
      vector<lower=0>[N] Y;
      real min_value;
    }

    parameters {
      real<lower=0> shape;
      real<lower=0> rate;
    }

    model {
      Y ~ gamma(shape, rate);
    }

    generated quantities {
      real mu = shape / rate + min_value;
      real mode = (shape - 1.0) / rate + min_value;
      
      vector[N] Y_s;
      for (n in 1:N) {
        Y_s[n] = gamma_rng(shape, rate) + min_value;
      }
    }
    """

    MIN_VALUE_BUFFER = 1.0e-10
    DEFAULT_ITERATION_COUNT = 1000
    DEFAULT_CHAINS = 3
    DEFAULT_WARMUP = 300

    def __init__(self):
        self.model = stan.StanModel(model_code=self.MODEL_CODE)

    def resamplings(self, matrix):
        resampling_mode = []
        for _, items in matrix.iteritems():
            fit = self.resampling(items)
            mode = np.mean(fit.extract('mode')['mode'])
            resampling_mode.append(mode)

        return pd.DataFrame(np.array(resampling_mode).reshape(40, 40))

    def resampling(self, series: pd.Series, iterator=DEFAULT_ITERATION_COUNT, chains=DEFAULT_CHAINS, warmup=DEFAULT_WARMUP):
        item_size = len(series)
        min_value = series.min() - self.MIN_VALUE_BUFFER
        series -= min_value
        stan_data = {
            'N': item_size,
            'Y': series.values.tolist(),
            'min_value': min_value
        }

        return self.model.sampling(data=stan_data, iter=iterator, chains=chains, warmup=warmup)
