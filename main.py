"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Ethical Machine Learning applied to  data from Techo.org                                   -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: renattaGS                                                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/renattaGS/PAP-ML-ETICO                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import data
import functions
import visualizations

# data import
file_path = 'files/Datos_Techo_converted.xlsx'
file_path_l = 'files/Data_techo_llenado.csv'
file_path_sl = 'files/Data_techo_solollenado.csv'

data_techo = data.lectura_datos(file_path, 'Datos Techo', 'p1')
data_techo_llenado = pd.read_csv(file_path_l)
data_techo_sllenado = pd.read_csv(file_path_sl)

missing_tot = (data_techo.isnull().sum()).sum()
missing = (data_techo.isnull().sum())['p74']
percentage = missing/len(data_techo) * 100

print('Existe un total de ', missing_tot, ' datos faltantes. \n Estos datos '
                                          'se encuentran ubicados en la columna p74')
print('Estos datos representan el ', percentage, '% del total de registros en la base de datos.')


data_techo.drop(data_techo[data_techo['p74'].isnull()].index, inplace=True)

# visualización de p74
visualizations.plot_3_histogram(data_techo['p74'].to_numpy(), data_techo_llenado['p74'].to_numpy(),
                                data_techo_sllenado['p74'].to_numpy(), 'p74', 'Frecuencia',
                                'Comparación de p74', 'Datos originales',
                                'Datos originales más simulados', 'Datos simulados')

cols_pairplot = ['p13', 'p27', 'p65', 'p71', 'p128', 'p131']
visualizations.pairplot(data_techo_llenado, cols_pairplot, 'p131' )

#Parametros
parametros = pd.read_csv('files/parametros.csv')
parametros.drop(columns=parametros.columns[0], axis=1, inplace=True)
parametros.set_index('Datos', inplace=True)
display(parametros)

# Partición de datos
cols_strat = ['p13']

X_train_org, X_test_org, y_train_org, y_test_org = functions.train_test_split_strat(data_techo,
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


np.random.seed(123)
w = np.random.normal(loc=1e-3, scale=1e-6, size=(1, X_train_org.shape[1]))
epochs = 1000

w1, J1, J_t1 = functions.gd(X_train_org, y_train_org, X_test_org, y_test_org, w, parametros.loc['Original', 'Alpha'],
                            parametros.loc['Original', 'Lambda'], epochs)
w2, J2, J_t2 = functions.gd(X_train_llenado, y_train_llenado, X_test_llenado, y_test_llenado, w, parametros.loc['Llenados', 'Alpha'],
                            parametros.loc['Llenados', 'Lambda'], epochs)
w3, J3, J_t3 = functions.gd(X_train_sllenado, y_train_sllenado, X_test_sllenado, y_test_sllenado, w, parametros.loc['Solo llenado', 'Alpha'],
                            parametros.loc['Solo llenado', 'Lambda'], epochs)
w1 = w1.reshape(-1)
w2 = w2.reshape(-1)
w3 = w3.reshape(-1)

visualizations.plot3_cost(J1, J_t1, J2, J_t2, J3, J_t3)

p1 = functions.h(X_train_org, w1)
y_hat1 = functions.predict(p1, 0.5)

p2 = functions.h(X_train_llenado, w2)
y_hat2 = functions.predict(p2, 0.5)

p3 = functions.h(X_train_sllenado, w3)
y_hat3 = functions.predict(p3, 0.5)

V1 = functions.get_values(y_train_org.reshape(-1), y_hat1)
V2 = functions.get_values(y_train_llenado.reshape(-1), y_hat2)
V3 = functions.get_values(y_train_sllenado.reshape(-1), y_hat3)

medidas_d = pd.DataFrame(columns=['Precisión', 'Sensibilidad', 'Especificidad'], index=parametros.index)
medidas_d['Precisión'] = [functions.accuracy(i) for i in [V1, V2, V3]]
medidas_d['Sensibilidad'] = [functions.sensitivity(i) for i in [V1, V2, V3]]
medidas_d['Especificidad'] = [functions.specificity(i) for i in [V1, V2, V3]]

display(medidas_d)


