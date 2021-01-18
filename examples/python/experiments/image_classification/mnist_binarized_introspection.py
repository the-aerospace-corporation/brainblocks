import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from brainblocks.blocks import BlankBlock, PatternClassifier

from _helper import mkdir_p

results_path = 'mnist_binarized/'
mkdir_p(results_path + 'active_statelets/')

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
    
    fname = results_path + 'active_statelets/' + 's{:04d}_l{:d}'.format(s, s_label)

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

labels = (0,1,2,3,4,5,6,7,8,9)

num_statelets = 1000

classifier = PatternClassifier(
    labels=labels,
    num_s=num_statelets,
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


    # ========================================
    # decode the receptive fields for active statelets
    # =======================================
    if i % 100 == 0:
        print()
        print('receptive fields of active statelets for training label={}'.format(y_train[i]))
        decoding = classifier.decode_bits()

        for y in range(28):
            for x in range(28):
                i = y * 28 + x
                if decoding[i] == 1:
                    print('X', end=' ')
                else:
                    print('-', end=' ')
            print()

t1 = time.time()
train_time = t1 - t0

# ========================================
# decode what the classifier thinks of this particular class
# =======================================
s_labels = []
cs_bits = []
for s in range(num_statelets):
    s_label = classifier.get_statelet_label(s)
    cs = classifier.coincidence_set(s)

    s_labels.append(s_label)
    cs_bits.append(cs.bits)

    #plot_statelet(s, s_label, cs.bits)

for label_val in labels:

    # total input bits rounded up to 800 from 28*28=784
    label_decoded = np.zeros(800, dtype=np.bool)

    # union of all label's receptive fields
    for s in range(num_statelets):
        if s_labels[s] == label_val:
            label_decoded = np.logical_or(label_decoded, np.array(cs_bits[s], dtype=np.bool))

    print()
    print('union of learned receptive fields for label={}'.format(label_val))
    decoding = classifier.decode_bits()

    for y in range(28):
        for x in range(28):
            i = y * 28 + x
            #if decoding[i] == 1:
            if label_decoded[i] == 1:
                print('X', end=' ')
            else:
                print('-', end=' ')
        print()

# plot the receptive field of each statelet in the PC
#for s in range(num_statelets):
#    s_label = classifier.get_statelet_label(s)
#    cs = classifier.coincidence_set(s)
#    plot_statelet(s, s_label, cs.bits)

