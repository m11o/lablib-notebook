import numpy as np
import pandas as pd

import pystan as stan


class MonteCarloResamplingOperator:
    MODEL_CODE = """
    data {
      int N;
      vector<lower=0>[N] Y;
    }
    
    parameters {
      real<lower=0> shape;
      real<lower=0> rate;
    }
    
    transformed parameters {
      real mu = shape / rate;
    }
    
    model {
      Y ~ gamma(shape, rate);
    }
    """

    def __init__(self, shuffle_calculator):
        df_index = list(range(len(shuffle_calculator.shuffle_dfs)))
        df_columns = list(range(len(shuffle_calculator.shuffle_dfs[0].columns) ** 2))
        self.df = pd.DataFrame(index=df_index, columns=df_columns)

        self.shuffle_calculator = shuffle_calculator
        self.__unify_df()

        self.model = stan.StanModel(model_code=self.MODEL_CODE)

    def resampling(self, matrix):
        resampling_mu = np.array([])
        for _, items in matrix.iteritems():
            item_size = len(items)
            min_value = items.min()
            items -= min_value
            stan_data = {
                'N': item_size,
                'Y': items.values.tolist()
            }

            fit = self.model.sampling(data=stan_data, iter=3000, chains=3, warmup=1000)
            mu = np.mean(fit.extract('mu')['mu']) + min_value
            resampling_mu = np.append(resampling_mu, mu)

        return resampling_mu

    def __unify_df(self):
        for index, shuffle_df in enumerate(self.shuffle_calculator.shuffle_dfs):
            self.df.iloc[index, :] = shuffle_df.to_numpy().ravel()
