import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# %%
digits = datasets.load_digits()

#
# Tamaño del dataset
#
digits.images.shape
# %%
