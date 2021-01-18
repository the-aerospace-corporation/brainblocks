#!/usr/bin/python
# ==============================================================================
# run_classifier_comparison.py
# ==============================================================================
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_s_curve, make_swiss_roll, make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from brainblocks.tools import BBClassifier
from brainblocks.datasets.classification import make_box_data_grid, make_box_data_random

import faulthandler
faulthandler.enable()

# h = .05  # step size in the mesh
h = .02  # step size in the mesh
num_samples = 500
rand_seed = 42

color_maps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys']
color_list = ['r', 'b', 'g', 'orange', 'purple', 'grey']

names = [
    'BBClassifier',
    'Nearest Neighbors',
    'Decision Tree',
    'Neural Net',
    'Naive Bayes']

classifiers = [
    BBClassifier(num_epochs=3, use_normal_dist_bases=True,
                 use_evenly_spaced_periods=True, random_state=rand_seed),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5, random_state=rand_seed),
    MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(100, 100),
                  random_state=rand_seed),
    GaussianNB()]


noise = 0.1

# Generate separable 2 classes dataset
X, y = make_classification(n_samples=num_samples, n_features=2, n_redundant=0,
                           n_informative=2, random_state=rand_seed,
                           n_clusters_per_class=1, n_classes=2)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(low=0.0, high=noise, size=X.shape)
separable_2_classes = (X, y)

# Generate separable 3 classes dataset
X, y = make_classification(n_samples=num_samples, n_features=2, n_redundant=0,
                           n_informative=2, random_state=rand_seed,
                           n_clusters_per_class=1, n_classes=3)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(low=0.0, high=noise, size=X.shape)
separable_3_classes = (X, y)

# Generate separable 4 classes dataset
X, y = make_classification(n_samples=num_samples, n_features=2, n_redundant=0,
                           n_informative=2, random_state=rand_seed,
                           n_clusters_per_class=1, n_classes=4)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(low=0.0, high=noise, size=X.shape)
separable_4_classes = (X, y)

# Add separable 2, 3, and 4 datasets
datasets = [separable_2_classes, separable_3_classes, separable_4_classes]

# Add moons, circles, and box datasets
datasets += [
    make_moons(n_samples=num_samples, noise=0.3, random_state=rand_seed),
    make_circles(n_samples=num_samples, noise=0.2, factor=0.5,
                 random_state=rand_seed),
    make_box_data_random(n_samples=num_samples, min_val=-0.3, max_val=1.3,
                         stratify=True, shuffle=True),
    make_box_data_grid(h=0.05, min_val=-0.3, max_val=1.3, shuffle=True)]

# Add blobs datasets
for k in range(2, 5):
    X, y = make_blobs(n_samples=num_samples, n_features=2, centers=k,
                      random_state=rand_seed)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(low=0.0, high=3*noise, size=X.shape)
    datasets.append((X, y))

# Setup matplotlib figure
figure = plt.figure(figsize=(18, 18))
i = 1

# Iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    print('Data Set %d' % ds_cnt)

    # Preprocess dataset, split into training and test part
    X, y = ds
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=rand_seed)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Color map based on number of classes
    n_classes = np.unique(y).size
    cm_bright = ListedColormap(color_list[:n_classes])

    # Plot the dataset first
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title('Input data')

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # Iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        print('Train %s' % name)

        t0 = time.time()
        clf.fit(X_train, y_train)
        t1 = time.time()
        print('Time %fs with size %d' % ((t1 - t0), y_train.shape[0]))

        t0 = time.time()
        score = clf.score(X_test, y_test)
        t1 = time.time()
        print('Score:', score)
        print('Time %fs with size %d' % ((t1 - t0), y_test.shape[0]))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        t0 = time.time()
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        t1 = time.time()
        print('Decision Space')
        print('Time %fs with size %d' % ((t1 - t0), xx.size))

        # Zero the classes that are not the maximum probability
        new_Z = []
        for row in Z:
            maxIndex = np.argmax(row)
            tempRow = np.zeros(row.shape)
            tempRow[maxIndex] = row[maxIndex]
            new_Z.append(tempRow)
        Z = np.asarray(new_Z)

        for k in range(n_classes):
            Z_class = Z[:, k].reshape(xx.shape)

            class_colors = get_cmap(color_maps[k % n_classes], 100)
            class_colors = class_colors(np.linspace(0, 1, 100))
            class_colors[:, 3] = np.linspace(0, 1, 100)

            class_cmp = ListedColormap(class_colors)

            ax.contourf(xx, yy, Z_class, cmap=class_cmp)

        # Plot the test points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

        print()

    print('-----------', flush=True)
    print()

plt.tight_layout()
plt.savefig('classifier_comparison.png')
plt.close()

for clf in classifiers:
    del clf
