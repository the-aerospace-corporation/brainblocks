# templates
from brainblocks.templates import Classifier, AbnormalityDetector

# printing boolean arrays neatly
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=100,
                    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})



def test_classifier():
    c = Classifier(labels=(0, 1),
                   min_val=0.0, max_val=1.0, num_i=1024, num_ai=128,
                   num_s=512, num_as=8, pct_pool=0.8, pct_conn=0.5, pct_learn=0.25)

    train_data = [
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

    c.fit([train_data], train_labels)

    probs = c.predict([[0.0]])[0]
    actual_probs = np.array(probs)
    expect_probs = np.array([1.0, 0.0])
    np.testing.assert_array_equal(actual_probs, expect_probs)

    probs = c.predict([[1.0]])[0]
    actual_probs = np.array(probs)
    expect_probs = np.array([0.0, 1.0])
    np.testing.assert_array_equal(actual_probs, expect_probs)


def test_abnormality_detector():
    ad = AbnormalityDetector(min_val=0.0, max_val=1.0, num_i=1024, num_ai=128,
                             num_s=512, num_as=8, num_spc=10, num_dps=10,
                             num_rpd=12, d_thresh=6, pct_pool=0.8, pct_conn=0.5,
                             pct_learn=0.25)

    data = [
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

    scores = ad.compute([data])
    actual_scores = np.array(scores)
    
    np.testing.assert_array_equal(actual_scores, expect_scores)


if __name__ == "__main__":
    #test_classifier()
    test_abnormality_detector()