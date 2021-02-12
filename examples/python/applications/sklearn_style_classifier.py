# ==============================================================================
# sklearn_style_classifier.py
# ==============================================================================
from brainblocks.tools import BBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

h = .02  # step size in the mesh
num_samples = 500
rand_seed = 42
noise = 0.1

clf = BBClassifier(num_epochs=3, random_state=rand_seed)

X, y = make_classification(n_samples=num_samples, n_features=2, n_redundant=0, n_informative=2,
                           random_state=rand_seed, n_clusters_per_class=1, n_classes=2)

rng = np.random.RandomState(rand_seed)
X += 2 * rng.uniform(low=0.0, high=noise, size=X.shape)

# scale the data
X = MinMaxScaler().fit_transform(X)

# split data into train and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.2, random_state=rand_seed)

print("Training")
t0 = time.time()
clf.fit(X_train, y_train)
t1 = time.time()
print("Time %fs with size %d" % ((t1 - t0), y_train.shape[0]))

print("Predicting")
t0 = time.time()
score = clf.score(X_test, y_test)
t1 = time.time()
print("Time %fs with size %d" % ((t1 - t0), y_test.shape[0]))
print("Score:", score)
