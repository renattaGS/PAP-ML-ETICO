"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Ethical Machine Learning applied to  data from Techo.org                                   -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: renattaGS                                                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/renattaGS/PAP-ML-ETICO                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_discrete(distribucion: np.array, xlabel: str, ylabel: str, title: str):
    """Función para graficar el histograma de una distribución discreta"""
    plt.figure(figsize=[8, 4])
    plt.hist(distribucion, bins=len(set(distribucion)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
