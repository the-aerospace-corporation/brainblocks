# ==============================================================================
# anom_reset_sequence.py
# ==============================================================================
from brainblocks.blocks import LabelTransformer, SequenceLearner
from sklearn import preprocessing

# Example showing the use of reset function for sequence learning. Useful when
# learning batches of finite sequences instead of online learning of streaming
# data. Note, the first data point after reset is always abnormal since no
# predictions are made input is always unexpected and triggers a surprise update

values = [
    'a', 'a', 'a', 'a', 'a',
    'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a',
    'a', 'a', 'a', 'a', 'a',
    'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a',
    'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a']

scores = [0.0 for _ in range(len(values))]

# convert to integer values
le = preprocessing.LabelEncoder()
le.fit(values)
labels = le.transform(values)

# define blocks
lt = LabelTransformer(
    num_l=26,  # number of labels
    num_s=208) # number of statelets

sl = SequenceLearner(
    num_spc=10,  # number of statelets per column
    num_dps=10,  # number of dendrites per statelet
    num_rpd=12,  # number of receptors per dendrite
    d_thresh=6,  # dendrite threshold
    perm_thr=20, # receptor permanence threshold
    perm_inc=2,  # receptor permanence increment
    perm_dec=1)  # receptor permanence decrement

# Connect blocks
sl.input.add_child(lt.output, 0)

# Repeat 3 times
for iteration_i in range(3):

    # Loop through data
    for i in range(len(labels)):

        # Compute
        lt.set_value(labels[i])
        lt.feedforward()
        sl.feedforward(learn=True)

        # Get anomaly score
        scores[i] = sl.get_anomaly_score()

    # Print output
    print("iteration %d" % iteration_i)
    print("value, score")
    for i in range(len(scores)):
        print("%5s, %0.1f" % (values[i], scores[i]))
    print()
    scores = [0.0 for _ in range(len(values))]

    # Reset sequence by clearing the blocks
    lt.clear()
    sl.clear()
