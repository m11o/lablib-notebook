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
      vector<lower=0>[N] Y_s = gamma_rng(shape, rate);
    }
    """

    MIN_VALUE_BUFFER = 1.0e-10

    def __init__(self):
        self.model = stan.StanModel(model_code=self.MODEL_CODE)

    def resampling(self, matrix):
        resampling_mu = []
        for _, items in matrix.iteritems():
            item_size = len(items)
            min_value = items.min() - self.MIN_VALUE_BUFFER
            items -= min_value
            stan_data = {
                'N': item_size,
                'Y': items.values.tolist(),
                'min_value': min_value
            }

            fit = self.model.sampling(data=stan_data, iter=1000, chains=3, warmup=300)
            mu = np.mean(fit.extract('mu')['mu'])
            resampling_mu.append(mu)

        return pd.DataFrame(np.array(resampling_mu).reshape(40, 40))

    def __unify_df(self):
        for index, shuffle_df in enumerate(self.shuffle_calculator.shuffle_dfs):
            self.df.iloc[index, :] = shuffle_df.to_numpy().ravel()
