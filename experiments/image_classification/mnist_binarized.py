# ignore tensorflow warnings regarding numpy version
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from brainblocks.blocks import BlankBlock, PatternClassifier

# helper function to convert monochrome image pixels to bits
def binarize(pixels, threshold=128):
    return 1 * (pixels > threshold)

# helper function to flatten 2d image to 1d vector
def flatten(image):
    return [y for x in image for y in x] 

# retrieve MNIST data via Tensorflow
print("Loading MNIST data...", flush=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# define train/test parameters
num_epochs=1
num_trains=len(x_train)
num_tests=len(x_test)
pixel_thresh=128 # from 0 to 255

# setup BrainBlocks classifier architecture
blankblock = BlankBlock(num_s=784)

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

classifier.input.add_child(blankblock.output)

# train BrainBLocks classifier
print("Training...", flush=True)
t0 = time.time()
for _ in range(num_epochs):
    for i in range(num_trains):
        bitimage = binarize(x_train[i], pixel_thresh)
        blankblock.output.bits = flatten(bitimage)
        classifier.compute(y_train[i], learn=True)
t1 = time.time()
train_time = t1 - t0

# test BrainBLocks Classifier
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
t1 = time.time()
test_time = t1 - t0

# output results
accuracy = num_correct / num_tests
print("Results:")
print("- number of training images: {:d}".format(num_trains), flush=True)
print("- number of testing images: {:d}".format(num_tests), flush=True)
print("- training time: {:0.6f}s".format(train_time), flush=True)
print("- testing time: {:0.6f}s".format(test_time), flush=True)
print("- accuracy: {:0.2f}%".format(accuracy*100), flush=True)

# display sample image
#plt.subplot(121),plt.imshow(x_train[0], cmap='gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#bitimage = binarize(x_train[0], pixel_thresh)
#plt.subplot(122),plt.imshow(bitimage, cmap='gray')
#plt.title('Binariazed Image'), plt.xticks([]), plt.yticks([])
#plt.show()