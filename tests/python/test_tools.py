# ==============================================================================
# test_tools.py
# ==============================================================================
from brainblocks.tools import BBClassifier

# printing boolean arrays neatly
import numpy as np
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=100,
    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

rand_seed = 42

# ==============================================================================
# Test BBClassifier
# ==============================================================================
def test_bbclassifier():

    c = BBClassifier(
        num_epochs=3,
        use_normal_dist_bases=True,
        use_evenly_spaced_periods=True,
        random_state=rand_seed)

    train_data = np.array([
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).reshape(-1, 1)

    train_labels = np.array([
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).reshape(-1)

    c.fit(train_data, train_labels)

    preds = c.predict([[0.0]])
    actual_preds = np.array(preds)
    expect_preds = np.array([0])
    np.testing.assert_array_equal(actual_preds, expect_preds)

    preds = c.predict([[1.0]])
    actual_preds = np.array(preds)
    expect_preds = np.array([1])
    np.testing.assert_array_equal(actual_preds, expect_preds)

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":

    test_bbclassifier()
