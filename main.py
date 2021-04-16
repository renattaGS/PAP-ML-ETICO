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
import numpy as np
import data
import functions
import visualizations

# data import
file_path = 'files/Datos_Techo_converted.xlsx'

data_techo = data.lectura_datos(file_path, 'Datos Techo', 'p1')

missing_tot = (data_techo.isnull().sum()).sum()
missing = (data_techo.isnull().sum())['p74']

print('Existen un total de ', missing_tot, ' valores faltantes en el Data frame \n')
print('Existen un total de ', missing, 'valores faltnates en la columna p74')

# visualizaciónde p74
ne_data = data_techo[data_techo['p74'].notnull()]['p74'].to_numpy()
visualizations.plot_histogram_discrete(ne_data, 'p74', 'Frecuencia',
                                       'Histograma de datos no faltantes en p74')
# Datos faltantes

data_techo = functions.fill_empty(data_techo, 'p74', 'p3', 'p27')
missing_tot = (data_techo.isnull().sum()).sum()
print('Existen un total de ', missing_tot, ' valores faltantes en el Data frame')

visualizations.plot_histogram_discrete(data_techo['p74'].to_numpy(), 'p74', 'Frecuencia',
                                       'Histograma de p74')

# Reporte de datos
data.reporte_profiling(data_techo, 'Reporte_Datos_Techo.html')

# Partición de datos
cols_strat = ['p13']
X_train, X_test, y_train, y_test = functions.train_test_split_strat(data_techo,
                                                                    0.20, cols_strat)
features = X_train.columns.tolist()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

cols_estan = ['p14', 'p15', 'p19', 'p20', 'p26', 'p32', 'p33', 'p34', 'p35',
              'p36', 'p39', 'p40', 'p47', 'p66', 'p67', 'p74', 'p91', 'p92', 'p93',
              'p99', 'p101', 'p104']

z_train, z_test = functions.standarizacion(X_train, X_test, cols_estan)

X_train[cols_estan] = z_train
X_test[cols_estan] = z_test

X_train['bias'] = 1
x_train = X_train[['bias'] + features].values
y_train = y_train.values
y_train = y_train.reshape(y_train.shape[0], 1)

X_test['bias'] = 1
x_test = X_test[['bias'] + features].values
y_test = y_test.values
y_test = y_test.reshape(y_test.shape[0], 1)


print('train ', X_train.shape, y_train.shape)
print('test ', X_test.shape, y_test.shape)

np.random.seed(123)
w = np.random.normal(loc=1e-3, scale=1e-6, size=(1, x_train.shape[1]))
lmbd = 0.05
alpha = 0.005
epochs = 7000
print('w shape', w.shape)

print('train cost ', functions.cost(x_train, y_train, w, lmbd))
w, J, J_t = functions.gd(X_train, y_train, x_test, y_test, w, alpha, lmbd, epochs)
print('train cost ', functions.cost(X_train, y_train, w, lmbd))
w = w.reshape(-1)
print(w)

plt.plot(range(len(J)), J, label='Train cost')
plt.plot(range(len(J_t)), J_t, label='Test cost')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()

p = functions.h(x_train, w)  # get probabilities
y_hat = functions.predict(p, 0.5)
print(y_test[:20].squeeze())
print(y_hat[:20])

V = functions.get_values(y_train.reshape(-1), y_hat)
print('sensitivity', functions.sensitivity(V))
print('specificity', functions.specificity(V))

Se, Sp = functions.ROC(y_train, functions.h(x_train, w))
plt.plot(1 - Sp, Se, '-')

Se, Sp = functions.ROC(y_test, functions.h(x_test, w))
plt.plot(1 - Sp, Se, '-')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend(['Train', 'Test'])
plt.title('Receiver Operating Characteristic')
plt.show()

print('Train AUC: ', metrics.roc_auc_score(y_train, functions.h(x_train, w)))
print('Test AUC: ', metrics.roc_auc_score(y_test, functions.h(x_test, w)))
