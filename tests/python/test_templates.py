# ==============================================================================
# test_templates.py
# ==============================================================================
from brainblocks.templates import AnomalyDetector, AnomalyDetectorPersist, Classifier

# Printing boolean arrays neatly
import numpy as np
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=100,
    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

# ==============================================================================
# Test Anomaly Detector
# ==============================================================================
def test_anomaly_detector():

    ad = AnomalyDetector(
        min_val=0.0,   # minimum input value
        max_val=1.0,   # maximum input value
        num_i=1024,    # number of input statelets
        num_ai=128,    # number of active input statelets
        num_s=512,     # number of statelets
        num_as=8,      # number of active statelets
        num_spc=10,    # number of statelets per column
        num_dps=10,    # number of dendrites per statelet
        num_rpd=12,    # number of receptors per dendrite
        d_thresh=6,    # dendrite threshold
        pct_pool=0.8,  # pooling percentage
        pct_conn=0.5,  # initially connected percentage
        pct_learn=0.3) # learn percentage

    values = [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    actual_scores = [0.0 for _ in range(len(values))]

    expect_scores = np.array([
        1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(len(values)):
        actual_scores[i] = ad.feedforward(values[i], learn=True)

    np.testing.assert_array_equal(actual_scores, expect_scores)

# ==============================================================================
# Test Anomaly Detector Persist
# ==============================================================================
def test_anomaly_detector_persist():

    ad = AnomalyDetectorPersist(
        min_val=0.0,   # minimum input value
        max_val=1.0,   # maximum input value
        max_step=8,    # maximum persistence step
        num_i=1024,    # number of input statelets
        num_ai=128,    # number of active input statelets
        num_s=512,     # number of statelets
        num_as=8,      # number of active statelets
        num_spc=10,    # number of statelets per column
        num_dps=10,    # number of dendrites per statelet
        num_rpd=12,    # number of receptors per dendrite
        d_thresh=6,    # dendrite threshold
        pct_pool=0.8,  # pooling percentage
        pct_conn=0.5,  # initially connected percentage
        pct_learn=0.3) # learn percentage

    values = [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    actual_scores = [0.0 for _ in range(len(values))]

    expect_scores = np.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(len(values)):
        actual_scores[i] = ad.feedforward(values[i], learn=True)

    # TODO: verify proper functionality
    #np.testing.assert_array_equal(actual_scores, expect_scores)

# ==============================================================================
# Test Classifier
# ==============================================================================
def test_classifier():

    c = Classifier(
        num_l=2,       # number of labels
        min_val=0.0,   # minimum input value
        max_val=1.0,   # maximum input value
        num_i=1024,    # number of input statelets
        num_ai=128,    # number of active input statelets
        num_s=512,     # number of statelets
        num_as=8,      # number of active statelets
        pct_pool=0.8,  # pooling percentage
        pct_conn=0.5,  # initially connected percentage
        pct_learn=0.3) # learn percentage

    x_train = [
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    y_train = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    for i in range(len(x_train)):
        c.fit(x_train[i], y_train[i])

    prob = c.predict(0.0)
    actual_probs = np.array(prob)
    expect_probs = np.array([1.0, 0.0])
    np.testing.assert_array_equal(actual_probs, expect_probs)

    prob = c.predict(1.0)
    actual_probs = np.array(prob)
    expect_probs = np.array([0.0, 1.0])
    np.testing.assert_array_equal(actual_probs, expect_probs)

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":

    test_anomaly_detector()
    test_classifier()
