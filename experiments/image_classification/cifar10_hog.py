# ignore tensorflow warnings regarding numpy version
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.feature import hog
from brainblocks.blocks import BlankBlock, PatternClassifier

# helper function to convert HOG feature descriptor to bits
def binarize_hog(fd, threshold=0.3):
    return 1 * (fd > threshold)

# retrieve CIFAR-10 data via Tensorflow
print("Loading CIFAR-10 data...", flush=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# define train/test parameters
num_epochs=1
num_trains=len(x_train)
num_tests=len(x_test)
hog_thresh=0.25
orientations=9
pixels_per_cell=(4, 4)
cells_per_block=(2, 2)

hog_fd = hog(x_train[0],
             orientations=orientations,
             pixels_per_cell=pixels_per_cell,
             cells_per_block=cells_per_block,
             visualize=False,
             multichannel=True,
             feature_vector=True)

# setup BrainBlocks classifier architecture
input_block = BlankBlock(num_s=len(hog_fd))

classifier = PatternClassifier(
    labels=(0,1,2,3,4,5,6,7,8,9),
    num_s=8000,
    num_as=10,
    perm_thr=20,
    perm_inc=2,
    perm_dec=1,
    pct_pool=0.8,
    pct_conn=1.0,
    pct_learn=0.25)

classifier.input.add_child(input_block.output)

# train BrainBLocks classifier
print("Training...", flush=True)
t0 = time.time()
for _ in range(num_epochs):
    for i in range(num_trains):
        hog_fd = hog(x_train[i],
                     orientations=orientations,
                     pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block,
                     visualize=False,
                     multichannel=True,
                     feature_vector=True)
        bit_fd = binarize_hog(hog_fd, hog_thresh)
        input_block.output.bits = bit_fd
        classifier.compute(int(y_train[i]), learn=True)
t1 = time.time()
train_time = t1 - t0

# test BrainBLocks Classifier
print("Testing...", flush=True)
num_correct = 0
t0 = time.time()
for i in range(num_tests):
    hog_fd = hog(x_test[i],
                 orientations=orientations,
                 pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block,
                 visualize=False,
                 multichannel=True,
                 feature_vector=True)
    bit_fd = binarize_hog(hog_fd, hog_thresh)
    input_block.output.bits = bit_fd
    classifier.compute(learn=False)
    probs = classifier.get_probabilities()
    if np.argmax(probs) == y_test[i]:
        num_correct += 1
t1 = time.time()
test_time = t1 - t0

# output results
accuracy = num_correct / num_tests
print("Results:")
print("- number of training images: {:d}".format(num_trains), flush=True)
print("- number of testing images: {:d}".format(num_tests), flush=True)
print("- HOG feature descriptor size: {:d}".format(len(hog_fd)), flush=True)
print("- training time: {:0.6f}s".format(train_time), flush=True)
print("- testing time: {:0.6f}s".format(test_time), flush=True)
print("- accuracy: {:0.2f}%".format(accuracy*100), flush=True)