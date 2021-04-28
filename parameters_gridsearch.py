"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Ethical Machine Learning applied to  data from Techo.org                                   -- #
# -- script: missing_data.py : python script with the parameter grid search for the models               -- #
# -- author: renattaGS                                                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/renattaGS/PAP-ML-ETICO                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data
import functions
import pandas as pd
import numpy as np

file_path = 'files/Datos_Techo_converted.xlsx'
data_techo_org = data.lectura_datos(file_path, 'Datos Techo', 'p1')
data_techo_llenado = pd.read_csv('files/Data_techo_llenado.csv')
data_techo_sllenado = pd.read_csv('files/Data_techo_llenado.csv')

data_techo_org.drop(data_techo_org[data_techo_org['p74'].isnull()].index, inplace=True)

cols_strat = ['p13']

X_train_org, X_test_org, y_train_org, y_test_org = functions.train_test_split_strat(data_techo_org,
                                                                                    0.10, cols_strat)

X_train_llenado, X_test_llenado, y_train_llenado, y_test_llenado = functions.train_test_split_strat(data_techo_llenado,
                                                                                                    0.20, cols_strat)

X_train_sllenado, X_test_sllenado, y_train_sllenado, y_test_sllenado = functions.train_test_split_strat(data_techo_sllenado,
                                                                                                        0.10, cols_strat)
features = X_train_org.columns.tolist()

cols_estan = ['p14', 'p15', 'p19', 'p20', 'p26', 'p32', 'p33', 'p34', 'p35',
              'p36', 'p39', 'p40', 'p47', 'p66', 'p67', 'p74', 'p91', 'p92', 'p93',
              'p99', 'p101', 'p104']

z_train, z_test = functions.standarizacion(X_train_org, X_test_org, cols_estan)
X_train_org[cols_estan] = z_train
X_test_org[cols_estan] = z_test

z_train, z_test = functions.standarizacion(X_train_llenado, X_test_llenado, cols_estan)
X_train_llenado[cols_estan] = z_train
X_test_llenado[cols_estan] = z_test

z_train, z_test = functions.standarizacion(X_train_sllenado, X_test_sllenado, cols_estan)
X_train_sllenado[cols_estan] = z_train
X_test_sllenado[cols_estan] = z_test

X_train_org['bias'] = 1
X_train_org = X_train_org[['bias'] + features].values
y_train_org = y_train_org.values
y_train_org = y_train_org.reshape(y_train_org.shape[0], 1)

X_test_org['bias'] = 1
X_test_org = X_test_org[['bias'] + features].values
y_test_org = y_test_org.values
y_test_org = y_test_org.reshape(y_test_org.shape[0], 1)

X_train_llenado['bias'] = 1
X_train_llenado = X_train_llenado[['bias'] + features].values
y_train_llenado = y_train_llenado.values
y_train_llenado = y_train_llenado.reshape(y_train_llenado.shape[0], 1)

X_test_llenado['bias'] = 1
X_test_llenado = X_test_llenado[['bias'] + features].values
y_test_llenado = y_test_llenado.values
y_test_llenado = y_test_llenado.reshape(y_test_llenado.shape[0], 1)

X_train_sllenado['bias'] = 1
X_train_sllenado = X_train_sllenado[['bias'] + features].values
y_train_sllenado = y_train_sllenado.values
y_train_sllenado = y_train_sllenado.reshape(y_train_sllenado.shape[0], 1)

X_test_sllenado['bias'] = 1
X_test_sllenado = X_test_sllenado[['bias'] + features].values
y_test_sllenado = y_test_sllenado.values
y_test_sllenado = y_test_sllenado.reshape(y_test_sllenado.shape[0], 1)

w = np.random.normal(loc=1e-3, scale=1e-6, size=(1, X_train_org.shape[1]))
epochs = 100

lambda_org , alpha_org = functions.grid_search_lr(X_train_org, y_train_org,
                                                  X_test_org, y_test_org, w, epochs)

lambda_llenado , alpha_llenado = functions.grid_search_lr(X_train_llenado, y_train_llenado,
                                                  X_test_llenado, y_test_llenado, w, epochs)

lambda_sllenado , alpha_sllenado = functions.grid_search_lr(X_train_sllenado, y_train_sllenado,
                                                  X_test_sllenado, y_test_sllenado, w, epochs)

final_df = pd.DataFrame(columns=['Datos', 'Lambda', 'Alpha'])
final_df['Datos'] = ['Original', 'Llenados', 'Solo llenado']
final_df['Lambda'] = [lambda_org, lambda_llenado, lambda_sllenado]
final_df['Alpha'] = [alpha_org, alpha_llenado, alpha_sllenado]

data.descarga_datos(final_df, 'parametros')