import warnings
warnings.filterwarnings('ignore') # ignore tensorflow warnings about numpy version
import time
import numpy as np
np.set_printoptions(threshold=100000)
import tensorflow as tf
from brainblocks.blocks import BlankBlock, PatternClassifier
from _helper import mkdir_p, binarize, flatten, plot_iteration

results_path = 'mnist_binarized/'
mkdir_p(results_path)
mkdir_p(results_path + 'results/')
mkdir_p(results_path + 'active_statelets/')

# retrieve MNIST data
print("Loading MNIST data...", flush=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# define train/test parameters
num_epochs=1
num_trains=len(x_train)
num_tests=len(x_test)
pixel_thresh=128 # from 0 to 255

# setup BrainBlocks architecture
blankblock = BlankBlock(num_s=784)
classifier = PatternClassifier(
    labels=(0,1,2,3,4,5,6,7,8,9),
    num_s=784, #8000
    num_as=9, #10
    perm_thr=20,
    perm_inc=2,
    perm_dec=1,
    pct_pool=0.8,
    pct_conn=1.0,
    pct_learn=0.25)

classifier.input.add_child(blankblock.output)

# train BrainBlocks classifier
print("Training...", flush=True)
t0 = time.time()
for _ in range(num_epochs):
    for i in range(num_trains):
        bitimage = binarize(x_train[i], pixel_thresh)
        blankblock.output.bits = flatten(bitimage)
        classifier.compute(y_train[i], learn=True)
t1 = time.time()
train_time = t1 - t0

# test BrainBlocks classifier
print("Testing...", flush=True)
num_correct = 0
t0 = time.time()
for i in range(num_tests):
    bitimage = binarize(x_test[i], pixel_thresh)
    blankblock.output.bits = flatten(bitimage)
    classifier.compute(learn=False)
    probs = classifier.get_probabilities()
    if np.argmax(probs) == y_test[i]:
        num_correct += 1

    if i < 100:
        plot_iteration(results_path, i, y_test[i], x_test[i], bitimage, classifier)

t1 = time.time()
test_time = t1 - t0

# output results
accuracy = num_correct / num_tests
print("Results:")
print("- number of training images: {:d}".format(num_trains))
print("- number of testing images: {:d}".format(num_tests))
print("- training time: {:0.6f}s".format(train_time))
print("- testing time: {:0.6f}s".format(test_time))
print("- accuracy: {:0.2f}%".format(accuracy*100))