import numpy as np
import pandas as pd

import pystan as stan


class MonteCarloResamplingOperator:
    MODEL_CODE = """
    data {
      int N;
      vector[N] Y;
    }
    
    parameters {
      real mu;
      real<lower=0> sigma;
    }
    
    model {
      for (n in 1:N) {
        Y[N] ~ normal(mu, sigma);
      }
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
        for index, items in matrix.iteritems():
            stan_data = {
                'N': len(items),
                'Y': items.values.tolist()
            }

            fit = self.model.sampling(data=stan_data, iter=3000, chains=3, warmup=1000)
            mu = np.mean(fit.extract('mu')['mu'])
            print(mu)
            resampling_mu = np.append(resampling_mu, mu)

        return resampling_mu

    def sampling(self, matrix):
        stan_data = {
            'N': len(matrix),
            'M': len(matrix.columns),
            'y': matrix.values.tolist()
        }

        return self.model.sampling(data=stan_data, iter=1000, chains=2, warmup=300)

    def __unify_df(self):
        for index, shuffle_df in enumerate(self.shuffle_calculator.shuffle_dfs):
            self.df.iloc[index, :] = shuffle_df.to_numpy().ravel()
