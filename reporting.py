"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Ethical Machine Learning applied to  data from Techo.org                                   -- #
# -- script: reporting.py: python script for the creation of Pandas profiling report                     -- #
# -- author: renattaGS                                                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/renattaGS/PAP-ML-ETICO                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data
import pandas as pd

file_path = 'files/Datos_Techo_converted.xlsx'
data_techo_org = data.lectura_datos(file_path, 'Datos Techo', 'p1')
data_techo_llenado = pd.read_csv('files/Data_techo_llenado.csv')
data_techo_sllenado = pd.read_csv('files/Data_techo_solollenado.csv')

data_techo_org.drop(data_techo_org[data_techo_org['p74'].isnull()].index, inplace=True)

data.reporte_profiling(data_techo_org, 'Reporte_Datos_Techo.html')
data.reporte_profiling(data_techo_llenado, 'Reporte_Datos_Techo_llenado.html')
data.reporte_profiling(data_techo_sllenado, 'Reporte_Datos_Techo_sllenado.html')