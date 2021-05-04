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


def acep_rechazo_simplificada_dis(N: int, Dom_f: tuple, f, max_f: float) -> np.array:
    X = np.zeros(N)
    i = 0
    while i < N:
        R2 = np.random.uniform(0, max_f)
        R1 = np.random.randint(Dom_f[0], Dom_f[1])
        if R2 <= f([R1]):
            X[i] = R1
            i += 1
    return X


def fill_empty(df: pd.DataFrame, col_empty: str, col_criteria1: str, col_criteria2: str):
    warnings.filterwarnings('ignore')
    unique_c1 = df[col_criteria1].unique().tolist()
    unique_c2 = df[col_criteria2].unique().tolist()

    not_empty = df[df[col_empty].notnull()]
    empty = df[df[col_empty].isnull()]
    aux_df = pd.DataFrame(columns=df.columns)

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

                for k, l in zip(idx_aux_e, range(len(idx_aux_e))):
                    df.loc[k, col_empty] = acep_rechazo_simplificada_dis(1, (min_empty, max_empty), gen_pdf, max(pdf))
                    aux_df.loc[l, :] = df.loc[k, :]
    df.drop(df[df[col_empty].isnull()].index, inplace=True)
    return df, aux_df


def train_test_split_strat(data: pd.DataFrame, test_size: float,
                           strat_cols: list) -> np.array:
    col_drop = ['p1', 'p3', 'p4', 'p5', 'p6', 'p7', 'p131']
    x = data.drop(col_drop, axis=1)
    y = data['p131']
    strat_data = data[strat_cols]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=143,
                                                        shuffle=True,
                                                        stratify=strat_data)
    return X_train, X_test, y_train, y_test


def standarizacion(x_train: np.array, x_test: np.array, feats: list) -> np.array:
    mu_train, std_train = x_train[feats].mean(axis=0), x_train[feats].std(axis=0)
    mu_test, std_test = x_test[feats].mean(axis=0), x_test[feats].std(axis=0)
    z_train = (x_train[feats] - mu_train) / std_train
    z_test = (x_test[feats] - mu_test) / std_test

    return z_train, z_test


def h(x: np.array, w: np.array) -> np.array:
    """Sigmoide function"""
    z_i = np.dot(x, w.T)
    sigma = 1 / (1 + np.exp(-z_i))
    return sigma


def cost(x: np.array, y: np.array, w: np.array, lmbd: float) -> np.array:
    m = len(y)
    j_w_1 = (1 - y) * np.log(1 - h(x, w))
    j_w_2 = y * np.log(h(x, w))
    j_w_3 = j_w_2 + j_w_1
    j_w = -1 * np.average(j_w_3)
    j_reg = (lmbd / (2 * m)) * np.dot(w, w.T).squeeze()
    fin = j_w + j_reg
    return fin


def grad(x: np.array, y: np.array, w: np.array, lmbd: float) -> np.array:
    m = len(y)
    d_j_w_1 = 1 / m
    d_j_w_2 = np.dot(x.T, (h(x, w) - y))
    d_j_w = (d_j_w_1 * d_j_w_2)
    d_j_w_reg = np.sum((lmbd * w) / m)
    fin = d_j_w + d_j_w_reg
    return fin


def gd(x: np.array, y: np.array, x_t: np.array, y_t: np.array,
       w: np.array, alpha: float, lmbd: float, epochs: int):
    J = []  # loss function with training set
    J_t = []  # loss function with test set
    for i in range(epochs):
        w_grad = grad(x, y, w, lmbd).T  # compute the gradinet
        w = w - alpha * w_grad  # update weights
        J.append(cost(x, y, w, lmbd))  # compute the cost for training
        J_t.append(cost(x_t, y_t, w, lmbd))  # compute the cost for testing
    return w, J, J_t


def predict(p: np.array, thr: float) -> np.array:
    """
    p: is the probability (hypothesis)
    thr: is the threshold
    return: 0 or 1 for each instance
    """
    indx = p > thr
    p[indx] = 1
    p[~indx] = 0
    return p.astype(int)


def get_values(y: np.array, y_hat: np.array) -> dict:
    """
    y: the true values
    y_hat: the predicted values
    return: A dictionary
    """
    V = {}
    V['tp'] = sum(y[y_hat == 1] == 1)
    V['fn'] = sum(y[y_hat == 0] == 1)
    V['fp'] = sum(y[y_hat == 1] == 0)
    V['tn'] = sum(y[y_hat == 0] == 0)
    return V


def sensitivity(V: dict) -> float:
    return V['tp'] / (V['tp'] + V['fn'])


def specificity(V: dict) -> float:
    return V['tn'] / (V['tn'] + V['fp'])


def accuracy(V: dict) -> float:
    return (V['tp'] + V['tn']) / (V['tp'] + V['tn'] + V['fp'] + V['fn'])


def ROC(y: np.array, p_roc: np.array):
    """
    Compute sensitivity and specificity for each threshold in [0, 1]
    return: Se, Sp
    """
    thrs = np.arange(0, 1, 0.05)
    Se = np.zeros((len(thrs)))
    Sp = np.zeros((len(thrs)))
    for idx, thr in enumerate(thrs):
        y_hat = predict(p_roc.copy(), thr)
        V = get_values(y.reshape(-1), y_hat)
        Se[idx] = sensitivity(V)
        Sp[idx] = specificity(V)
    return Se, Sp


def grid_search_lr(X_train, y_train, x_test, y_test, w, epochs):
    grid = np.arange(0.001, 1, 0.001)
    opt_j = 1
    opt_jt = 1
    opt_lmbd = 0
    opt_alpha = 0
    for lmbd in grid:
        for alpha in grid:
            w, J, J_t = gd(X_train, y_train, x_test, y_test, w, alpha, lmbd, epochs)
            if (np.mean(J) < opt_j) & (np.mean(J_t) < opt_jt):
                opt_j = np.mean(J)
                opt_jt = np.mean(J_t)
                opt_lmbd = lmbd
                opt_alpha = alpha
    return opt_lmbd, opt_alpha
