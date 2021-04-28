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
import pandas as pd
import numpy as np
import seaborn as sns


def plot_histogram_discrete(distribucion: np.array, xlabel: str, ylabel: str, title: str):
    """Función para graficar el histograma de una distribución discreta"""
    plt.figure(figsize=[8, 4])
    plt.hist(distribucion, bins=len(set(distribucion)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_3_histogram(dist1: np.array, dist2: np.array, dist3: np.array,
                     xlabel: str, ylabel: str, title: str, title1: str, title2: str, title3: str):
    fig, axs = plt.subplots(3, figsize=(15, 15))
    fig.suptitle(title)
    axs[0].hist(dist1, bins=len(set(dist1)))
    axs[0].set_title(title1)
    axs[1].hist(dist2, bins=len(set(dist2)))
    axs[1].set_title(title2)
    axs[2].hist(dist3, bins=len(set(dist3)))
    axs[2].set_title(title3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def pairplot(df: pd.DataFrame, cols: list, hue: str):
    plot = df[cols]
    sns.pairplot(plot, hue=hue)


def plot3_cost(train1: np.array, test1: np.array, train2: np.array, test2: np.array, train3: np.array, test3: np.array):
    fig, axs = plt.subplots(3, figsize=(15, 15))
    fig.suptitle('Funciones de costo')
    axs[0].plot(range(len(train1)), train1, label='Costo Entrenamiento')
    axs[0].plot(range(len(test1)), test1, label='Costo Entrenamiento')
    axs[0].set_title('Funciones de costo 1')
    axs[0].legend()
    axs[1].plot(range(len(train2)), train2, label='Costo Entrenamiento')
    axs[1].plot(range(len(test2)), test2, label='Costo Prueba')
    axs[1].set_title('Funciones de costo 2')
    axs[1].legend()
    axs[2].plot(range(len(train3)), train3, label='Costo Entrenamiento')
    axs[2].plot(range(len(test3)), test3, label='Costo Prueba')
    axs[2].set_title('Funciones de costo 3')
    axs[2].legend()
    plt.xlabel('Iterations')
    plt.ylabel('Iterations')
