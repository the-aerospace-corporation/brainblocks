# ==============================================================================
# online_learning_discrete_sequence_anomalies.py
# ==============================================================================
from brainblocks.blocks import DiscreteTransformer, SequenceLearner
from sklearn import preprocessing

train_values = [
    'a', 'a', 'a', 'a', 'a', 'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a', 'b', 'c', 'd', 'e', 'f']

test_values = [
    'a', 'a', 'a', 'a', 'a', 'b', 'c', 'g', 'e', 'f']

train_scores = [0.0 for _ in range(len(train_values))]
test_scores = [0.0 for _ in range(len(test_values))]

# Convert values to integers
le = preprocessing.LabelEncoder()
le.fit(train_values+test_values)
train_integers = le.transform(train_values)
test_integers = le.transform(test_values)

# Setup blocks
lt = DiscreteTransformer(
    num_v=26,  # number of discrete values
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

# Loop through the values twice
for iteration_i in range(2):
    for i in range(len(train_integers)):

        # Set scalar transformer value
        lt.set_value(train_integers[i])

        # Compute the scalar transformer
        lt.feedforward()

        # Compute the sequence learner w/ learning
        sl.feedforward(learn=True)

        # Get anomaly score
        train_scores[i] = sl.get_anomaly_score()

    # Reset SequenceLearner
    lt.clear()
    sl.clear()

# Reset SequenceLearner
lt.clear()
sl.clear()

# Loop through the values
for i in range(len(test_integers)):

    # Set scalar transformer value
    lt.set_value(test_integers[i])

    # Compute the scalar transformer
    lt.feedforward()

    # Compute the sequence learner w/o learning
    sl.feedforward(learn=False)

    # Get anomaly score
    test_scores[i] = sl.get_anomaly_score()

# Print output
print("TRAIN")
print("val, scr")
for i in range(len(train_values)):
    print("%3s, %0.1f" % (train_values[i], train_scores[i]))

# Print output
print()
print("TEST")
print("val, scr")
for i in range(len(test_values)):
    print("%3s, %0.1f" % (test_values[i], test_scores[i]))
