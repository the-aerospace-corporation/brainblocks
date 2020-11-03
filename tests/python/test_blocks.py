from brainblocks.blocks import *
from sklearn import preprocessing
import os
import numpy as np

# printing boolean arrays neatly
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=100,
                    formatter={'bool': lambda bin_val: 'X' if bin_val else '-'})

# setup constants
CURR = 0
PREV = 1

# ==============================================================================
# Test Read/Write Statelets
# ==============================================================================
def test_read_write_statelets():
    blank = BlankBlock(num_s=32)

    wbits = [
        1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]

    wacts = [28, 29, 30, 31]

    # setting bits
    blank.output[CURR].bits = wbits

    # getting bits
    rbits = blank.output[CURR].bits

    # setting acts
    blank.output[CURR].acts = wacts

    # getting acts
    racts = blank.output[CURR].acts

    np.testing.assert_array_equal(np.array(wbits), np.array(rbits))
    np.testing.assert_array_equal(np.array(wacts), np.array(racts))

# ==============================================================================
# Read Coincidence Set
# ==============================================================================
def test_read_coincidence_set():
    e = ScalarEncoder()
    pp = PatternPooler()

    pp.input.add_child(e.output)
    pp.initialize()

    cs = pp.output_coincidence_set(0)

    addrs = cs.get_addrs()
    addr0 = cs.get_addr(0)
    np.testing.assert_equal(addrs[0], addr0)

    perms = cs.get_perms()
    perm0 = cs.get_perm(0)
    np.testing.assert_equal(perms[0], perm0)

# ==============================================================================
# ScalarEncoder
# ==============================================================================
def test_scalar_encoder():
    e = ScalarEncoder(
        min_val=-1.0, # minimum input value
        max_val=1.0,  # maximum input value
        num_s=1024,   # number of statelets
        num_as=128)   # number of active statelets

    e.compute(value=-1.5)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=-1.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=-0.5)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[224:352] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[448:576] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.5)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[672:800] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=1.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=1.5)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

# ==============================================================================
# SymbolsEncoder
# ==============================================================================
def test_symbols_encoder():
    # Symbols as strings, converted to integers
    le = preprocessing.LabelEncoder()
    expect_symbols = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    le.fit(expect_symbols)
    int_symbols = le.transform(expect_symbols)

    e = SymbolsEncoder(
        max_symbols=8, # maximum number of symbols
        num_s=1024)    # number of statelets

    e.compute(value=int_symbols[0])
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=int_symbols[1])
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[128:256] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=int_symbols[2])
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[256:384] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=int_symbols[3])
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[384:512] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=int_symbols[4])
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[512:640] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=int_symbols[5])
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[640:768] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=int_symbols[6])
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[768:896] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=int_symbols[7])
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    # C++ side will throw a warning here
    #e.compute(value=int_symbols[8]) // TODO: put old code back here

# ==============================================================================
# PersistenceEncoder
# ==============================================================================
def test_persistence_encoder():
    e = PersistenceEncoder(
        min_val=-1.0, # minimum input value
        max_val=1.0,  # maximum input value
        num_s=1024,   # number of statelets
        num_as=128,   # number of active statelets
        max_steps=4)  # maximum number of persistence steps

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[224:352] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[448:576] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[672:800] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    e.compute(value=0.0)
    actual_bits = np.array(e.output[CURR].bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

# ==============================================================================
# PatternClassifier
# ==============================================================================
def test_pattern_classifier():
    e = SymbolsEncoder(
        max_symbols=8, # maximum number of symbols
        num_s=1024)    # number of statelets

    pc = PatternClassifier(
        labels=(0, 1),  # user-defined labels
        num_s=512,      # number of statelets
        num_as=8,       # number of active statelets
        perm_thr=20,    # receptor permanence threshold
        perm_inc=2,     # receptor permanence increment
        perm_dec=1,     # receptor permanence decrement
        pct_pool=0.8,   # pooling percentage
        pct_conn=0.5,   # initially connected percentage
        pct_learn=0.25) # learn percentage

    pc.input.add_child(e.output)

    for _ in range(10):
        e.compute(value=0)
        pc.compute(label=0, learn=True)
        e.compute(value=1)
        pc.compute(label=1, learn=True)

    e.compute(value=0)
    pc.compute(learn=False)
    probs = pc.get_probabilities()
    labels = pc.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 0
    np.testing.assert_equal(actual_label, expect_label)

    e.compute(value=1)
    pc.compute(learn=False)
    probs = pc.get_probabilities()
    labels = pc.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 1
    np.testing.assert_equal(actual_label, expect_label)

# ==============================================================================
# PatternClassifierDynamic
# ==============================================================================
def test_pattern_classifier_dynamic():
    e = SymbolsEncoder(
        max_symbols=8, # maximum number of symbols
        num_s=1024)    # number of statelets

    pc = PatternClassifierDynamic(
        num_s=512,      # number of statelets
        num_as=8,       # number of active statelets
        num_spl=32,      # number of statelets per label
        perm_thr=20,    # receptor permanence threshold
        perm_inc=2,     # receptor permanence increment
        perm_dec=1,     # receptor permanence decrement
        pct_pool=0.8,   # pooling percentage
        pct_conn=0.5,   # initially connected percentage
        pct_learn=0.25) # learn percentage

    pc.input.add_child(e.output)

    for _ in range(10):
        e.compute(value=0)
        pc.compute(label=0, learn=True)
        e.compute(value=1)
        pc.compute(label=1, learn=True)

    e.compute(value=0)
    pc.compute(learn=False)
    probs = pc.get_probabilities()
    labels = pc.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 0
    np.testing.assert_equal(actual_label, expect_label)

    e.compute(value=1)
    pc.compute(learn=False)
    probs = pc.get_probabilities()
    labels = pc.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 1
    np.testing.assert_equal(actual_label, expect_label)

# ==============================================================================
# PatternPooler
# ==============================================================================
def test_pattern_pooler():
    e = SymbolsEncoder(
        max_symbols=8, # maximum number of symbols
        num_s=1024)    # number of statelets

    pp = PatternPooler(
        num_s=512,      # number of statelets
        num_as=8,       # number of active statelets
        perm_thr=20,    # receptor permanence threshold
        perm_inc=2,     # receptor permanence increment
        perm_dec=1,     # receptor permanence decrement
        pct_pool=0.8,   # pooling percentage
        pct_conn=0.5,   # initially connected percentage
        pct_learn=0.25) # learn percentage

    pp.input.add_child(e.output)

    e.compute(value=0)
    pp.compute(learn=False)
    before_a = pp.output[CURR].bits

    e.compute(value=1)
    pp.compute(learn=False)
    before_b = pp.output[CURR].bits

    for _ in range(10):
        e.compute(value=0)
        pp.compute(learn=True)
        e.compute(value=1)
        pp.compute(learn=True)

    e.compute(value=0)
    pp.compute(learn=False)
    after_a = pp.output[CURR].bits

    e.compute(value=1)
    pp.compute(learn=False)
    after_b = pp.output[CURR].bits

    #np.testing.assert_array_equal(before_a, after_a) # TODO: figure out how to verify... random start location causes 1 or 2 statelets to change
    #np.testing.assert_array_equal(before_b, after_b)

    #pp.save(file_str='pp.bin')
    #pp.load(file_str='pp.bin')
    #os.remove('pp.bin')

# ==============================================================================
# SequenceLearner Square
# ==============================================================================
def test_sequence_learner_square():
    e = ScalarEncoder(
        min_val=0.0, # minimum input value
        max_val=1.0, # maximum input value
        num_s=64,    # number of statelets
        num_as=8)    # number of active statelets

    sl = SequenceLearner(
        num_spc=10, # number of statelets per column
        num_dps=10, # number of coincidence detectors per statelet
        num_rpd=12, # number of receptors per coincidence detector
        d_thresh=6, # coincidence detector threshold
        perm_thr=1, # receptor permanence threshold
        perm_inc=1, # receptor permanence increment
        perm_dec=0) # receptor permanence decrement

    sl.input.add_child(e.output)

    values = [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    actual_scores = np.array([0.0 for i in range(len(values))])

    expect_scores = np.array([
        1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(len(values)):
        e.compute(value=values[i])
        sl.compute(learn=True)
        actual_scores[i] = sl.get_score()

    np.testing.assert_array_equal(actual_scores, expect_scores)

# ==============================================================================
# SequenceLearner Triangle
# ==============================================================================
def test_sequence_learner_triangle():
    e = ScalarEncoder(
        min_val=0.0, # minimum input value
        max_val=1.0, # maximum input value
        num_s=64,    # number of statelets
        num_as=8)    # number of active statelets

    sl = SequenceLearner(
        num_spc=10, # number of statelets per column
        num_dps=10, # number of coincidence detectors per statelet
        num_rpd=12, # number of receptors per coincidence detector
        d_thresh=6, # coincidence detector threshold
        perm_thr=1, # receptor permanence threshold
        perm_inc=1, # receptor permanence increment
        perm_dec=0) # receptor permanence decrement

    sl.input.add_child(e.output)

    values = [
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]

    actual_scores = np.array([0.0 for i in range(len(values))])

    expect_scores = np.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(len(values)):
        e.compute(value=values[i])
        sl.compute(learn=True)
        actual_scores[i] = sl.get_score()

    np.testing.assert_array_equal(actual_scores, expect_scores)

# ==============================================================================
# SequenceLearner Sine
# ==============================================================================
def test_sequence_learner_sine():
    e = ScalarEncoder(
        min_val=0.0, # minimum input value
        max_val=1.0, # maximum input value
        num_s=64,    # number of statelets
        num_as=8)    # number of active statelets

    sl = SequenceLearner(
        num_spc=10, # number of statelets per column
        num_dps=10, # number of coincidence detectors per statelet
        num_rpd=12, # number of receptors per coincidence detector
        d_thresh=6, # coincidence detector threshold
        perm_thr=1, # receptor permanence threshold
        perm_inc=1, # receptor permanence increment
        perm_dec=0) # receptor permanence decrement

    sl.input.add_child(e.output)

    values = [
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21,
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21,
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21]

    actual_scores = np.array([0.0 for i in range(len(values))])

    expect_scores = np.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(len(values)):
        e.compute(value=values[i])
        sl.compute(learn=True)
        actual_scores[i] = sl.get_score()

    np.testing.assert_array_equal(actual_scores, expect_scores)

'''
# ==============================================================================
# SequenceLearner Save Load
# ==============================================================================
def test_sequence_learner_save_load():
    e0 = ScalarEncoder(min_val=0.0, max_val=1.0, num_s=64, num_as=8)
    e1 = ScalarEncoder(min_val=0.0, max_val=1.0, num_s=64, num_as=8)
    sl0 = SequenceLearner(num_spc=10, num_dps=10, num_rpd=12, d_thresh=6, perm_thr=1, perm_inc=1, perm_dec=0)
    sl1 = SequenceLearner(num_spc=10, num_dps=10, num_rpd=12, d_thresh=6, perm_thr=1, perm_inc=1, perm_dec=0)
    sl0.input.add_child(e0.output)
    sl1.input.add_child(e1.output)

    values = [
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21,
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21,
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21]

    scores = np.array([0.0 for i in range(len(values))])
    expect_scores = np.array([0.0 for i in range(len(values))])
    expect_scores[0] = 1.0

    # save model from sl0
    for i in range(len(values)):
        e0.compute(value=values[i])
        sl0.compute(learn=True)

    sl0.save(file_str='sl.bin')

    # load model into sl1

    sl1.load(file_str='sl.bin')

    for i in range(len(values)):
        e1.compute(value=values[i])
        sl1.compute(learn=True)
        scores[i] = sl1.get_score()

    np.testing.assert_array_equal(scores, expect_scores)
    
    os.remove('sl.bin')
'''

# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    test_read_write_statelets()
    test_read_coincidence_set()
    test_scalar_encoder()
    test_symbols_encoder()
    test_persistence_encoder()
    test_pattern_classifier()
    test_pattern_classifier_dynamic()
    test_pattern_pooler()
    test_sequence_learner_square()
    test_sequence_learner_triangle()
    test_sequence_learner_sine()
    #test_sequence_learner_save_load()