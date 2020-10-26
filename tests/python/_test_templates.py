from brainblocks.templates import Classifier, AbnormalityDetector

# printing boolean arrays neatly
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=100,
                    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})


# ==============================================================================
# Classifier
# ==============================================================================
def test_classifier():
    c = Classifier(
        labels=(0, 1),  # user-defined labels
        min_val=0.0,    # minimum input value
        max_val=1.0,    # maximum input value
        num_i=1024,     # number of input statelets
        num_ai=128,     # number of active input statelets
        num_s=512,      # number of statelets
        num_as=8,       # number of active statelets
        pct_pool=0.8,   # pooling percentage
        pct_conn=0.5,   # initially connected percentage
        pct_learn=0.25) # learn percentage

    train_values = [
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0]

    train_labels = [
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1]

    c.fit([train_values], train_labels)

    probs = c.predict([[0.0]])[0]
    actual_probs = np.array(probs)
    expect_probs = np.array([1.0, 0.0])
    np.testing.assert_array_equal(actual_probs, expect_probs)

    probs = c.predict([[1.0]])[0]
    actual_probs = np.array(probs)
    expect_probs = np.array([0.0, 1.0])
    np.testing.assert_array_equal(actual_probs, expect_probs)

# ==============================================================================
# Classifier
# ==============================================================================
def test_abnormality_detector():
    ad = AbnormalityDetector(
        min_val=0.0,    # minimum input value
        max_val=1.0,    # maximum input value
        num_i=1024,     # number of input statelets
        num_ai=128,     # number of active input statelets
        num_s=512,      # number of statelets
        num_as=8,       # number of active statelets
        num_spc=10,     # number of statelets per column
        num_dps=10,     # number of coincidence detectors per statelet
        num_rpd=12,     # number of receptors per coincidence detector
        d_thresh=6,     # coincidence detector threshold
        pct_pool=0.8,   # pooling percentage
        pct_conn=0.5,   # initially connected percentage
        pct_learn=0.25) # learn percentage

    values = [
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0]

    expect_scores = np.array([
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0])

    scores = ad.compute([values])
    actual_scores = np.array(scores)
    
    np.testing.assert_array_equal(actual_scores, expect_scores)

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    test_classifier()
    test_abnormality_detector()