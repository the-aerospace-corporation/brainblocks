# ==============================================================================
# hypergrid_transformer.py
# ==============================================================================
from brainblocks.tools import HyperGridTransform
from sklearn.datasets import make_classification
import numpy as np

# printing boolean arrays neatly
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=80,
                    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

# generate 5-dimensional data set
X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=4,
                           n_clusters_per_class=2, flip_y=0.01, class_sep=3.0,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True,
                           random_state=42)

# create hypergrid transform
# num_grids * num_bins = total_bits
hgt = HyperGridTransform(num_grids=10, num_bins=8, num_subspace_dims=1)

# fit the data
hgt.fit(X)

# transform scalar feature vectors to distributed binary representation
bits_tensor = hgt.transform(X)

for i in range(len(X)):
    print(X[i])
    print(bits_tensor[i])
