# ==============================================================================
# mnist_binarized.py
# ==============================================================================
import warnings
warnings.filterwarnings('ignore') # ignore tensorflow warnings on numpy version
import time
import numpy as np
import tensorflow as tf
from brainblocks.blocks import BlankBlock, PatternClassifier
from _helper import mkdir_p, binarize, flatten, plot_iteration

results_path = 'mnist_binarized/'
mkdir_p(results_path + 'results/')
mkdir_p(results_path + 'active_statelets/')

# Retrieve MNIST data
print("Loading MNIST data...", flush=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

run_t0 = time.time()

# Define scenario parameters
num_epochs=1
num_trains=len(x_train)
num_tests=len(x_test)
pixel_thresh=128 # from 0 to 255
num_s = 8000

# Setup BrainBlocks architecture
blankblock = BlankBlock(num_s=784)
classifier = PatternClassifier(
    num_l=10,
    num_s=num_s,
    num_as=9,
    perm_thr=20,
    perm_inc=2,
    perm_dec=1,
    pct_pool=0.8,
    pct_conn=1.0,
    pct_learn=0.3)

classifier.input.add_child(blankblock.output, 0)

# Train BrainBlocks classifier
bb_train_time = 0
print("Training...", flush=True)

for _ in range(num_epochs):
    for i in range(num_trains):
        bitimage = binarize(x_train[i], pixel_thresh)
        blankblock.output.bits = flatten(bitimage)
        blankblock.feedforward()
        t0 = time.time()
        classifier.set_label(y_train[i])
        classifier.feedforward(learn=True)
        t1 = time.time()
        bb_train_time += t1 - t0

# Test BrainBlocks classifier
num_error = 0
bb_test_time = 0
print("Testing...", flush=True)

for i in range(num_tests):
    bitimage = binarize(x_test[i], pixel_thresh)
    blankblock.output.bits = flatten(bitimage)
    blankblock.feedforward()
    t0 = time.time()
    classifier.feedforward(learn=False)
    probs = classifier.get_probabilities()
    if np.argmax(probs) != y_test[i]:
        num_error += 1
    t1 = time.time()
    bb_test_time += t1 - t0

    if i < 100:
        plot_iteration(results_path, i, y_test[i], x_test[i], bitimage, classifier, num_s)

run_t1 = time.time()

# Output results
accuracy = 1 - (num_error / num_tests)
print("Results:")
print("- Number of training images: {:d}".format(num_trains))
print("- Number of testing images: {:d}".format(num_tests))
print("- Total run time: {:0.6f}s".format(run_t1 - run_t0))
print("- BB training time: {:0.6f}s".format(bb_train_time))
print("- BB testing time: {:0.6f}s".format(bb_test_time))
print("- Accuracy: {:0.2f}%".format(accuracy*100))
