# ==============================================================================
# SequenceLearner Save and Load
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, SequenceLearner
import numpy as np
import os

values = [
    0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21,
    0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21,
    0.50, 0.79, 0.98, 0.98, 0.79, 0.50, 0.21, 0.02, 0.02, 0.21]

# array for anomaly scores
scores = np.array([0.0 for i in range(len(values))])

# expect all zeroes for learned sequence except for first value
expect_scores = np.array([0.0 for i in range(len(values))])
expect_scores[0] = 1.0

st0 = ScalarTransformer()
sl0 = SequenceLearner()
sl0.input.add_child(st0.output, 0)

# train sequence learner
for i in range(len(values)):
    st0.set_value(values[i])
    st0.feedforward()
    sl0.feedforward(learn=True)

# Save sequence learner memories
sl0.save('sl.bin')

# load trained SequenceLearner from file
st1 = ScalarTransformer()
sl1 = SequenceLearner()
sl1.input.add_child(st1.output, 0)

# load sequence learner memories
sl1.load('sl.bin')
os.remove('sl.bin')

# compute anomaly scores
for i in range(len(values)):
    st1.set_value(values[i])
    st1.feedforward()
    sl1.feedforward(learn=False)
    scores[i] = sl1.get_anomaly_score()

np.testing.assert_array_equal(scores, expect_scores)

# Print output
print("Expected and Actual Anomaly Scores")
print("val, exp, act")
for i in range(len(values)):
    print("%0.1f, %0.1f, %0.1f" % (values[i], expect_scores[i], scores[i]))


