import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from brainblocks.blocks import BlankBlock, PatternClassifier

# helper function to convert monochrome image pixels to bits
def binarize_image(pixels, threshold=128):
    return 1 * (pixels > threshold)

# helper function to flatten 2d image to 1d vector
def flatten_image(image):
    return [y for x in image for y in x] 

def plot(actual_img, backtrace, actual, predicted):
    back_img = [[0 for x in range(28)] for y in range(28)]
    for y in range(28):
        for x in range(28):
            i = x + y * 28
            back_img[y][x] = backtrace[i]
            #print(backtrace[i], end='')
        #print()
    #print()

    plt.subplot(121),plt.imshow(actual_img, cmap='gray')
    plt.title('Actual=' + str(actual)), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(back_img, cmap='gray')
    plt.title('Predicted=' + str(predicted)), plt.xticks([]), plt.yticks([])
    plt.show()

def plot_statelet(s, s_label, bits):
    bits_img = [[0 for x in range(28)] for y in range(28)]
    for y in range(28):
        for x in range(28):
            i = x + y * 28
            bits_img[y][x] = bits[i]
    
    fname = 's{:04d}_l{:d}'.format(s, s_label)

    plt.imshow(bits_img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fname, facecolor='gray', bbox_inches='tight', dpi=40)
    plt.close()

# retrieve MNIST data via Tensorflow
print("Loading MNIST data...", flush=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# define train/test parameters
num_trains = len(x_train)
num_tests = len(x_test)
pixel_thresh = 128 # from 0 to 255

# setup BrainBlocks classifier architecture
input_block = BlankBlock(num_s=784)

classifier = PatternClassifier(
    labels=(0,1,2,3,4,5,6,7,8,9),
    num_s=1000,
    num_as=10,
    perm_thr=20,
    perm_inc=2,
    perm_dec=1,
    pct_pool=0.1,
    pct_conn=1.0,
    pct_learn=0.25)

classifier.input.add_child(input_block.output)

# train BrainBLocks classifier
print("Training...", flush=True)
t0 = time.time()
for i in range(num_trains):
    bitimage = binarize_image(x_train[i], pixel_thresh)
    input_block.output.bits = flatten_image(bitimage)
    classifier.compute(y_train[i], learn=True)
t1 = time.time()
train_time = t1 - t0

for s in range(1000):
    s_label = classifier.get_statelet_label(s)
    cs = classifier.coincidence_set(s)
    plot_statelet(s, s_label, cs.bits)
'''
# test BrainBLocks Classifier
print("Testing...", flush=True)
num_correct = 0
t0 = time.time()
for i in range(num_tests):
    bitimage = binarize_image(x_test[i], pixel_thresh)
    input_block.output.bits = flatten_image(bitimage)
    classifier.compute(0, learn=False)
    probs = classifier.get_probabilities()
    predicted = np.argmax(probs)
    actual = y_test[i]
    if predicted == actual:
        num_correct += 1
    #else:
    #    backtrace = classifier.get_backtrace_bits()
    #    plot(x_test[i], backtrace, actual, predicted)
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
'''
# display sample image
#plt.subplot(121),plt.imshow(x_test[1], cmap='gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#bitimage = binarize_image(x_test[1], pixel_thresh)
#plt.subplot(122),plt.imshow(bitimage, cmap='gray')
#plt.title('Binariazed Image'), plt.xticks([]), plt.yticks([])
#plt.show()


# ========================================
# decode example
# ========================================
print()
print('label={}'.format(y_test[-1]))
#s_acts = classifier.output.acts
#cs = classifier.coincidence_set(s_acts[0])
#cs_bits = cs.bits
decoding = classifier.decode_bits()

for y in range(28):
    for x in range(28):
        i = y * 28 + x
        if decoding[i] == 1:
            print('X', end=' ')
        else:
            print('-', end=' ')
    print()
