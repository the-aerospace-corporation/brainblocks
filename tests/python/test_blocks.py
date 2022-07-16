# ==============================================================================
# test_blocks.py
# ==============================================================================
from brainblocks.blocks import *
import os
import numpy as np

# printing boolean arrays neatly
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=100,
    formatter={'bool': lambda bin_val: 'X' if bin_val else '-'})

# setup constants
CURR = 0
PREV = 1

# ==============================================================================
# Test Read and Write Statelets
# ==============================================================================
def test_read_write_statelets():

    blank = BlankBlock(num_s=32, num_t=2)

    bits_clear = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    wbits_prev = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    wbits_curr = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    acts_clear = []

    wacts_prev = [0, 1, 2, 3]

    wacts_curr = [4, 5, 6, 7]

    # Initial State
    # -------------

    # Get bits and acts from state and histories
    rbits = blank.output.bits
    racts = blank.output.acts
    rbits_prev = blank.output[PREV].bits
    racts_prev = blank.output[PREV].acts
    rbits_curr = blank.output[CURR].bits
    racts_curr = blank.output[CURR].acts

     # Test bits and acts
    np.testing.assert_array_equal(np.array(rbits), np.array(bits_clear))
    np.testing.assert_array_equal(np.array(racts), np.array(acts_clear))
    np.testing.assert_array_equal(np.array(rbits_prev), np.array(bits_clear))
    np.testing.assert_array_equal(np.array(racts_prev), np.array(acts_clear))
    np.testing.assert_array_equal(np.array(rbits_curr), np.array(bits_clear))
    np.testing.assert_array_equal(np.array(racts_curr), np.array(acts_clear))

    # First Step
    # ----------

     # Set and get bits from state
    wbits = wbits_prev
    blank.output.bits = wbits # set bits
    rbits = blank.output.bits # get bits

    # Set and get acts from state
    wacts = wacts_prev
    blank.output.acts = wacts # set acts
    racts = blank.output.acts # get acts

    blank.feedforward()

   # Get bits and acts from state and histories
    rbits_prev = blank.output[PREV].bits
    racts_prev = blank.output[PREV].acts
    rbits_curr = blank.output[CURR].bits
    racts_curr = blank.output[CURR].acts

     # Test bits and acts
    np.testing.assert_array_equal(np.array(rbits), np.array(wbits_prev))
    np.testing.assert_array_equal(np.array(racts), np.array(wacts_prev))
    np.testing.assert_array_equal(np.array(rbits_prev), np.array(bits_clear))
    np.testing.assert_array_equal(np.array(racts_prev), np.array(acts_clear))
    np.testing.assert_array_equal(np.array(rbits_curr), np.array(wbits_prev))
    np.testing.assert_array_equal(np.array(racts_curr), np.array(wacts_prev))

    # Second Step
    # -----------

     # Set and get bits from state
    wbits = wbits_curr
    blank.output.bits = wbits # set bits
    rbits = blank.output.bits # get bits

    # Set and get acts from state
    wacts = wacts_curr
    blank.output.acts = wacts # set acts
    racts = blank.output.acts # get acts

    blank.feedforward()

   # Get bits and acts from state and histories
    rbits_prev = blank.output[PREV].bits
    racts_prev = blank.output[PREV].acts
    rbits_curr = blank.output[CURR].bits
    racts_curr = blank.output[CURR].acts

     # Test bits and acts
    np.testing.assert_array_equal(np.array(rbits), np.array(wbits_curr))
    np.testing.assert_array_equal(np.array(racts), np.array(wacts_curr))
    np.testing.assert_array_equal(np.array(rbits_prev), np.array(wbits_prev))
    np.testing.assert_array_equal(np.array(racts_prev), np.array(wacts_prev))
    np.testing.assert_array_equal(np.array(rbits_curr), np.array(wbits_curr))
    np.testing.assert_array_equal(np.array(racts_curr), np.array(wacts_curr))

# ==============================================================================
# Test Read Memories
# ==============================================================================
def test_read_memories():

    st = ScalarTransformer()
    pp = PatternPooler()

    pp.input.add_child(st.output, 0)
    pp.init()

    addrs0 = pp.memory.addrs(0)
    perms0 = pp.memory.perms(0)
    conns0 = pp.memory.conns(0)

    addrs1 = pp.memory.addrs(1)
    perms1 = pp.memory.perms(1)
    conns1 = pp.memory.conns(1)

    addrs_not_equal = np.any(np.not_equal(np.array(addrs0), np.array(addrs1)))
    np.testing.assert_equal(addrs_not_equal, True)

    np.testing.assert_array_equal(np.array(perms0), np.array(perms1))

    conns_not_equal = np.any(np.not_equal(np.array(conns0), np.array(conns1)))
    np.testing.assert_equal(conns_not_equal, True)

# ==============================================================================
# Test ContextLearner
# ==============================================================================
def test_context_learner():

    lt0 = DiscreteTransformer(
        num_v=8,  # number of discrete values
        num_s=64) # number of statelets

    lt1 = DiscreteTransformer(
        num_v=8,  # number of discrete values
        num_s=64) # number of statelets

    sl = SequenceLearner(
        num_spc=10, # number of statelets per column
        num_dps=10, # number of coincidence detectors per statelet
        num_rpd=12, # number of receptors per coincidence detector
        d_thresh=6, # coincidence detector threshold
        perm_thr=1, # receptor permanence threshold
        perm_inc=1, # receptor permanence increment
        perm_dec=0) # receptor permanence decrement

    sl.input.add_child(lt0.output, 0)
    sl.context.add_child(lt1.output, 0)

    lt0.set_value(0)
    lt1.set_value(0)
    lt0.feedforward()
    lt1.feedforward()
    sl.feedforward(learn=True)

    lt0.set_value(0)
    lt1.set_value(1)
    lt0.feedforward()
    lt1.feedforward()
    sl.feedforward(learn=True)

# ==============================================================================
# DiscreteTransformer
# ==============================================================================
def test_label_encoder():

    lt = DiscreteTransformer(
        num_v=8,    # number of discrete values
        num_s=1024, # number of statelets
        num_t=2)    # number of BlockOutput time steps (optional)

    lt.set_value(0)
    lt.feedforward()
    actual_bits = np.array(lt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    lt.set_value(1)
    lt.feedforward()
    actual_bits = np.array(lt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[128:256] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    lt.set_value(2)
    lt.feedforward()
    actual_bits = np.array(lt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[256:384] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    lt.set_value(3)
    lt.feedforward()
    actual_bits = np.array(lt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[384:512] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    lt.set_value(4)
    lt.feedforward()
    actual_bits = np.array(lt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[512:640] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    lt.set_value(5)
    lt.feedforward()
    actual_bits = np.array(lt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[640:768] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    lt.set_value(6)
    lt.feedforward()
    actual_bits = np.array(lt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[768:896] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    lt.set_value(7)
    lt.feedforward()
    actual_bits = np.array(lt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

# ==============================================================================
# Test PatternClassifier
# ==============================================================================
def test_pattern_classifier():

    lt = DiscreteTransformer(
        num_v=8,    # number of discrete values
        num_s=1024) # number of statelets

    pc = PatternClassifier(
        num_l=2,       # number of labels
        num_s=512,     # number of statelets
        num_as=8,      # number of active statelets
        perm_thr=20,   # receptor permanence threshold
        perm_inc=2,    # receptor permanence increment
        perm_dec=1,    # receptor permanence decrement
        pct_pool=0.8,  # pooling percentage
        pct_conn=0.5,  # initially connected percentage
        pct_learn=0.3) # learn percentage

    pc.input.add_child(lt.output, 0)

    for _ in range(10):

        lt.set_value(0)
        pc.set_label(0)
        lt.feedforward()
        pc.feedforward(learn=True)

        lt.set_value(1)
        pc.set_label(1)
        lt.feedforward()
        pc.feedforward(learn=True)

    lt.set_value(0)
    lt.feedforward()
    pc.feedforward(learn=False)
    probs = pc.get_probabilities()
    labels = pc.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 0
    np.testing.assert_equal(actual_label, expect_label)

    lt.set_value(1)
    lt.feedforward()
    pc.feedforward(learn=False)
    probs = pc.get_probabilities()
    labels = pc.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 1
    np.testing.assert_equal(actual_label, expect_label)

# ==============================================================================
# Test PatternClassifierDynamic
# ==============================================================================
def test_pattern_classifier_dynamic():

    lt = DiscreteTransformer(
        num_v=8,    # number of discrete values
        num_s=1024) # number of statelets

    pc = PatternClassifierDynamic(
        num_s=512,     # number of statelets
        num_as=8,      # number of active statelets
        num_spl=32,    # number of statelets per label
        perm_thr=20,   # receptor permanence threshold
        perm_inc=2,    # receptor permanence increment
        perm_dec=1,    # receptor permanence decrement
        pct_pool=0.8,  # pooling percentage
        pct_conn=0.5,  # initially connected percentage
        pct_learn=0.3) # learn percentage

    pc.input.add_child(lt.output, 0)

    for _ in range(10):

        lt.set_value(0)
        pc.set_label(0)
        lt.feedforward()
        pc.feedforward(learn=True)

        lt.set_value(1)
        pc.set_label(1)
        lt.feedforward()
        pc.feedforward(learn=True)

    lt.set_value(0)
    lt.feedforward()
    pc.feedforward(learn=False)
    probs = pc.get_probabilities()
    labels = pc.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 0
    np.testing.assert_equal(actual_label, expect_label)

    lt.set_value(1)
    lt.feedforward()
    pc.feedforward(learn=False)
    probs = pc.get_probabilities()
    labels = pc.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 1
    np.testing.assert_equal(actual_label, expect_label)

# ==============================================================================
# Test PatternPooler
# ==============================================================================
def test_pattern_pooler():

    lt = DiscreteTransformer(
        num_v=8,    # number of discrete values
        num_s=1024) # number of statelets

    pp = PatternPooler(
        num_s=512,     # number of statelets
        num_as=8,      # number of active statelets
        perm_thr=20,   # receptor permanence threshold
        perm_inc=2,    # receptor permanence increment
        perm_dec=1,    # receptor permanence decrement
        pct_pool=0.8,  # pooling percentage
        pct_conn=0.5,  # initially connected percentage
        pct_learn=0.3) # learn percentage

    pp.input.add_child(lt.output, 0)

    lt.set_value(0)
    lt.feedforward()
    pp.feedforward(learn=False)
    before_a = pp.output.bits

    lt.set_value(1)
    lt.feedforward()
    pp.feedforward(learn=False)
    before_b = pp.output.bits

    for _ in range(10):
        lt.set_value(0)
        lt.feedforward()
        pp.feedforward(learn=True)
        lt.set_value(1)
        lt.feedforward()
        pp.feedforward(learn=True)

    lt.set_value(0)
    lt.feedforward()
    pp.feedforward(learn=False)
    after_a = pp.output.bits

    lt.set_value(1)
    lt.feedforward()
    pp.feedforward(learn=False)
    after_b = pp.output.bits

    np.testing.assert_array_equal(before_a, after_a)
    np.testing.assert_array_equal(before_b, after_b)

# ==============================================================================
# Test PersistenceTransformer
# ==============================================================================
def test_persistence_encoder():

    pt = PersistenceTransformer(
        min_val=-1.0, # minimum input value
        max_val=1.0,  # maximum input value
        num_s=1024,   # number of statelets
        num_as=128,   # number of active statelets
        max_step=4,  # maximum number of persistence steps
        num_t=2)      # number of BlockOutput time steps (optional)

    pt.set_value(0.0)
    pt.feedforward()
    actual_bits = np.array(pt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    pt.set_value(0.0)
    pt.feedforward()
    actual_bits = np.array(pt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[224:352] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    pt.set_value(0.0)
    pt.feedforward()
    actual_bits = np.array(pt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[448:576] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    pt.set_value(0.0)
    pt.feedforward()
    actual_bits = np.array(pt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[672:800] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    pt.set_value(0.0)
    pt.feedforward()
    actual_bits = np.array(pt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    pt.set_value(0.0)
    pt.feedforward()
    actual_bits = np.array(pt.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

# ==============================================================================
# Test ScalarTransformer
# ==============================================================================
def test_scalar_transformer():

    st = ScalarTransformer(
        min_val=-1.0, # minimum input value
        max_val=1.0,  # maximum input value
        num_s=1024,   # number of statelets
        num_as=128,   # number of active statelets
        num_t=2)      # number of BlockOutput time steps (optional)

    st.set_value(-1.5)
    st.feedforward()
    actual_bits = np.array(st.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    st.set_value(-1.0)
    st.feedforward()
    actual_bits = np.array(st.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[0:128] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    st.set_value(-0.5)
    st.feedforward()
    actual_bits = np.array(st.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[224:352] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    st.set_value(0.0)
    st.feedforward()
    actual_bits = np.array(st.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[448:576] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    st.set_value(0.5)
    st.feedforward()
    actual_bits = np.array(st.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[672:800] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    st.set_value(1.0)
    st.feedforward()
    actual_bits = np.array(st.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

    st.set_value(1.5)
    st.feedforward()
    actual_bits = np.array(st.output.bits)
    expect_bits = np.array([0 for i in range(1024)])
    expect_bits[896:1024] = 1
    np.testing.assert_array_equal(actual_bits, expect_bits)

# ==============================================================================
# Test SequenceLearner Square
# ==============================================================================
def test_sequence_learner_square():

    st = ScalarTransformer(
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

    sl.input.add_child(st.output, 0)

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
        st.set_value(values[i])
        st.feedforward()
        sl.feedforward(learn=True)
        actual_scores[i] = sl.get_anomaly_score()

    np.testing.assert_array_equal(actual_scores, expect_scores)

# ==============================================================================
# Test SequenceLearner Triangle
# ==============================================================================
def test_sequence_learner_triangle():

    st = ScalarTransformer(
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

    sl.input.add_child(st.output, 0)

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
        st.set_value(values[i])
        st.feedforward()
        sl.feedforward(learn=True)
        actual_scores[i] = sl.get_anomaly_score()

    np.testing.assert_array_equal(actual_scores, expect_scores)

# ==============================================================================
# Test SequenceLearner Sine
# ==============================================================================
def test_sequence_learner_sine():

    st = ScalarTransformer(
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

    sl.input.add_child(st.output, 0)

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
        st.set_value(values[i])
        st.feedforward()
        sl.feedforward(learn=True)
        actual_scores[i] = sl.get_anomaly_score()

    np.testing.assert_array_equal(actual_scores, expect_scores)

# ==============================================================================
# SequenceLearner Save and Load
# ==============================================================================
def test_sequence_learner_save_load():
    st0 = ScalarTransformer()
    st1 = ScalarTransformer()
    sl0 = SequenceLearner()
    sl1 = SequenceLearner()
    sl0.input.add_child(st0.output, 0)
    sl1.input.add_child(st1.output, 0)

    values = [
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21,
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21,
        0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21]

    scores = np.array([0.0 for i in range(len(values))])
    expect_scores = np.array([0.0 for i in range(len(values))])
    expect_scores[0] = 1.0

    # Compute sl0
    for i in range(len(values)):
        st0.set_value(values[i])
        st0.feedforward()
        sl0.feedforward(learn=True)

    # Save sl0 memories
    sl0.save('sl.bin')

    # load sl0 memories into sl1
    sl1.load('sl.bin')

    # Compute sl1 using new memories
    for i in range(len(values)):
        st1.set_value(values[i])
        st1.feedforward()
        sl1.feedforward(learn=False)
        scores[i] = sl1.get_anomaly_score()

    np.testing.assert_array_equal(scores, expect_scores)

    os.remove('sl.bin')


# ==============================================================================
# SequenceLearner Save and Load
# ==============================================================================
def test_pattern_classifier_save_load():



    lt0 = DiscreteTransformer(
        num_v=8,    # number of discrete values
        num_s=1024) # number of statelets
    lt1 = DiscreteTransformer(
        num_v=8,    # number of discrete values
        num_s=1024) # number of statelets

    pc0 = PatternClassifier(
        num_l=2,       # number of labels
        num_s=512,     # number of statelets
        num_as=8,      # number of active statelets
        perm_thr=20,   # receptor permanence threshold
        perm_inc=2,    # receptor permanence increment
        perm_dec=1,    # receptor permanence decrement
        pct_pool=0.8,  # pooling percentage
        pct_conn=0.5,  # initially connected percentage
        pct_learn=0.3) # learn percentage

    pc1 = PatternClassifier(
        num_l=2,       # number of labels
        num_s=512,     # number of statelets
        num_as=8,      # number of active statelets
        perm_thr=20,   # receptor permanence threshold
        perm_inc=2,    # receptor permanence increment
        perm_dec=1,    # receptor permanence decrement
        pct_pool=0.8,  # pooling percentage
        pct_conn=0.5,  # initially connected percentage
        pct_learn=0.3) # learn percentage

    pc0.input.add_child(lt0.output, 0)
    pc1.input.add_child(lt1.output, 0)



    # Train
    for _ in range(10):

        lt0.set_value(0)
        pc0.set_label(0)
        lt0.feedforward()
        pc0.feedforward(learn=True)

        lt0.set_value(1)
        pc0.set_label(1)
        lt0.feedforward()
        pc0.feedforward(learn=True)


    # Test Success
    lt0.set_value(0)
    lt0.feedforward()
    pc0.feedforward(learn=False)
    probs = pc0.get_probabilities()
    labels = pc0.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 0
    np.testing.assert_equal(actual_label, expect_label)

    # Save sl0 memories
    pc0.save('pc.bin')

    # load sl0 memories into sl1
    pc1.load('pc.bin')

    # Test Success of Reloaded PC
    lt1.set_value(1)
    lt1.feedforward()
    pc1.feedforward(learn=False)
    probs = pc1.get_probabilities()
    labels = pc1.get_labels()
    actual_label = np.array(labels[np.argmax(probs)])
    expect_label = 1
    np.testing.assert_equal(actual_label, expect_label)

    os.remove('pc.bin')


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':

    test_read_write_statelets()
    test_read_memories()
    test_context_learner()
    test_label_encoder()
    test_pattern_classifier()
    test_pattern_classifier_dynamic()
    test_pattern_pooler()
    test_persistence_encoder()
    test_scalar_transformer()
    test_sequence_learner_square()
    test_sequence_learner_triangle()
    test_sequence_learner_sine()
    test_sequence_learner_save_load()
    test_pattern_classifier_save_load()
