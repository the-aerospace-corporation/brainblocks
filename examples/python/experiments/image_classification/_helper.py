import os
import errno
import math
import numpy as np
import matplotlib.pyplot as plt
from brainblocks.blocks import PatternClassifier

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def binarize(pixels, threshold=128):
    # converts monochrome image pixels to bits
    return 1 * (pixels > threshold)

def flatten(image):
    # flattens 2d image to 1d vector
    return [y for x in image for y in x] 

def plot_example(fname, img_raw, img_binary, img_encoded, img_decoded, label, pred, prob):
    fig, axes = plt.subplots(1, 4, figsize=(8, 3))
    title = 'label={:d}, pred={:d}, prob={:0.2f}'.format(label, pred, prob)
    plt.suptitle(title, fontsize=18)
    for i in range(4):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    axes[0].set_title('Raw Data'), 
    axes[0].imshow(img_raw, cmap='gray')
    axes[1].set_title('Binary Input'), 
    axes[1].imshow(img_binary, cmap='gray')
    axes[2].set_title('State'), 
    axes[2].imshow(img_encoded, cmap='gray')
    axes[3].set_title('Decoding'), 
    axes[3].imshow(img_decoded, cmap='gray')
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(fname, bbox_inches='tight')

def plot_statelets(fname, s_dicts=[]):
    # plots statelet information from a list of dictionaries
    # each dictionary contains:
    # - index: statelet index integer
    # - conns: statelet connections array

    num_s = len(s_dicts)
    if num_s == 0:
        print('Warning: s_dicts number of elements is zero in plot_statelets()')
        return

    square_axes = math.ceil(math.sqrt(num_s))
    fig = plt.figure(figsize=(8,8))

    for s in range(num_s):
        index = s_dicts[s]['index']
        conns = s_dicts[s]['conns']
        num_c = len(conns)
        square = math.ceil(math.sqrt(len(conns)))
        img = [[0 for x in range(square)] for y in range(square)]

        for y in range(square):
            for x in range(square):
                c = x + y * square
                if c < num_c:
                    img[y][x] = conns[c]
                else:
                    img[y][x] = 0

        ax = fig.add_subplot(square_axes, square_axes, s+1)
        ax.set_title(str(index))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(img, cmap='gray')

    plt.subplots_adjust(hspace=0.25)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def plot_iteration(path, idx, label, raw, binary, classifier, num_s):
    num_ix = len(raw[0])
    num_iy = len(raw)
    num_i = num_ix * num_iy

    square = math.ceil(math.sqrt(num_s))
    num_sx = square
    num_sy = square
    num_s_ = square * square

    s_dicts = []
    encoded = [0 for _ in range(num_s_)]
    decoded = [0 for _ in range(num_i)]
    for act in classifier.output.acts:
        conns = classifier.memory.conns(act)
        s_dict = {}
        s_dict['index'] = act
        s_dict['conns'] = conns
        s_dicts.append(s_dict)

        encoded[act] = 1

        for i in range(num_i):
            if conns[i] == 1:
                decoded[i] += 1

    encoded = np.array(encoded).reshape((num_sx, num_sy))
    decoded = np.array(decoded).reshape((num_ix, num_iy))

    probs = classifier.get_probabilities()
    pred = np.argmax(probs)
    prob = probs[pred]

    header = 'i' + str(idx)
    plot_example(path + 'results/' + header + '_result.png', raw, binary, encoded, decoded, label, pred, prob)
    plot_statelets(path + 'active_statelets/' + header + '_active_statelets.png', s_dicts)