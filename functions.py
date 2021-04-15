"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Ethical Machine Learning applied to  data from Techo.org                                   -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: renattaGS                                                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/renattaGS/PAP-ML-ETICO                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pandas as pd


def kde_statsmodels_func(x: np.array):
    """Multivariate Kernel Density Estimation with Statsmodels returns a func"""
    kde = KDEMultivariate(x,
                          bw='cv_ml',
                          var_type='u')
    return lambda u: kde.pdf(u)


def kde_statsmodels_m(x: np.array, x_grid: np.array) -> np.array:
    """Multivariate Kernel Density Estimation with Statsmodels"""
    kde = KDEMultivariate(x,
                          bw='cv_ml',  # bandwidth * np.ones_like(x),
                          var_type='u')
    return kde.pdf(x_grid)


def acep_rechazo_simplificada_dis(N: int, Dom_f: tuple, f, max_f: float):
    X = np.zeros(N)
    i = 0
    while i < N:
        R2 = np.random.uniform(0, max_f)
        R1 = np.random.randint(Dom_f[0], Dom_f[1])
        if R2 <= f([R1]):
            X[i] = R1
            i += 1
    return X


def fill_empty(df: pd.DataFrame, col_empty: str, col_criteria1: str, col_criteria2: str) -> pd.DataFrame:
    warnings.filterwarnings('ignore')
    unique_c1 = df[col_criteria1].unique().tolist()
    unique_c2 = df[col_criteria2].unique().tolist()

    not_empty = df[df[col_empty].notnull()]
    empty = df[df[col_empty].isnull()]

    for i in unique_c1:
        for j in unique_c2:

            if empty[(empty[col_criteria1] == i) & (empty[col_criteria2] == j)][col_empty].empty:
                continue

            elif not_empty[(not_empty[col_criteria1] == i) & (not_empty[col_criteria2] == j)][col_empty].empty:
                continue

            else:

                aux_ne = (
                    not_empty[(not_empty[col_criteria1] == i) & (not_empty[col_criteria2] == j)][col_empty]).to_numpy()
                aux_e = (empty[(empty[col_criteria1] == i) & (empty[col_criteria2] == j)][col_empty])

                idx_aux_e = aux_e.index

                min_empty = min(df[df[col_empty].notnull()][col_empty])
                max_empty = max(df[df[col_empty].notnull()][col_empty])
                grid = np.arange(min_empty, max_empty)

                pdf = kde_statsmodels_m(aux_ne, grid, )
                gen_pdf = kde_statsmodels_func(aux_ne)

                for k in idx_aux_e:
                    df.loc[k, col_empty] = acep_rechazo_simplificada_dis(1, (min_empty, max_empty), gen_pdf, max(pdf))

    df.drop(df[df[col_empty].isnull()].index, inplace=True)
    return df


def train_test_split_strat(data: pd.DataFrame, test_size: float,
                           strat_cols: list):
    col_drop = ['p1', 'p3', 'p4', 'p5', 'p6', 'p7', 'p131']
    x = data.drop(col_drop, axis=1).to_numpy()
    y = data['p131'].to_numpy()
    strat_data = data[strat_cols].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True,
                                                        stratify=strat_data)
    return X_train, X_test, y_train, y_test
