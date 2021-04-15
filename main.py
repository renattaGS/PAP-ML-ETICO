"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Ethical Machine Learning applied to  data from Techo.org                                   -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: renattaGS                                                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/renattaGS/PAP-ML-ETICO                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
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
