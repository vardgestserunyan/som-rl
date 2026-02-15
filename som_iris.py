# Implement SOMs for the IRIS dataset
import sklearn.datasets as skdata
import numpy as np
import sklearn.preprocessing as skprep
from minisom import MiniSom
import matplotlib.pyplot as plt


iris_data = skdata.load_iris()
iris_x, iris_y, iris_names = iris_data['data'], iris_data['target'], iris_data['target_names']

iris_x = skprep.MinMaxScaler().fit_transform(iris_x)

som_size = 2
msom = MiniSom(som_size, som_size, 4, sigma=1, learning_rate=0.1, random_seed=12121995,\
               neighborhood_function='gaussian', activation_distance='euclidean',
               topology='rectangular')

msom.pca_weights_init(iris_x)
msom.train(iris_x, 1000)

som_map_dict = {}
for label in set(iris_y):
    iris_x_subset = iris_x[ iris_y == label ]
    counts_mat = np.zeros((som_size,som_size))
    winmap = msom.win_map(iris_x_subset)
    for cell, entries in winmap.items():
        x, y = cell[0], cell[1]
        counts_mat[x,y] = len(entries)
    som_map_dict[label] = counts_mat.copy()

fig, ax = plt.subplots(figsize=(9,3), nrows=1, ncols=len(set(iris_y)))
for key, value in som_map_dict.items():
    curr_ax = ax[int(key)]
    curr_ax.imshow(value, cmap='viridis')
    curr_ax.set_ylabel("Dim 2")
    curr_ax.set_xlabel("Dim 1")
    curr_ax.set_title(f"Self-Org Map for {iris_names[int(key)]}")
    curr_ax.set_yticks(range(som_size))
    curr_ax.set_xticks(range(som_size))




