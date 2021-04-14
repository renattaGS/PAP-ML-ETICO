"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Ethical Machine Learning applied to  data from Techo.org                                   -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: renattaGS                                                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/renattaGS/PAP-ML-ETICO                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
from pandas_profiling import ProfileReport


def lectura_datos(file_path: str, sheet: str, column_dateformat: str) -> pd.DataFrame:
    data = pd.read_excel(file_path, sheet)
    data[column_dateformat] = pd.to_datetime(data[column_dateformat], format='%Y').dt.year
    return data


def reporte_profiling(df: 'Data Frame', output_name: str):
    report = ProfileReport(df, minimal=True)
    report.to_file(output_file=output_name)
