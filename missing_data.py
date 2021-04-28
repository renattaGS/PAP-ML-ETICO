"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Ethical Machine Learning applied to  data from Techo.org                                   -- #
# -- script: missing_data.py : python script with the handling of missing data                           -- #
# -- author: renattaGS                                                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/renattaGS/PAP-ML-ETICO                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data
import functions


file_path = 'files/Datos_Techo_converted.xlsx'

data_techo = data.lectura_datos(file_path, 'Datos Techo', 'p1')

data_techo_nm, data_techo_om = functions.fill_empty(data_techo, 'p74', 'p3', 'p27')

data.descarga_datos(data_techo_nm, 'Data_techo_llenado')
data.descarga_datos(data_techo_om, 'Data_techo_solollenado')
