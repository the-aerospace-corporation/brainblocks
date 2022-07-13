# ==============================================================================
# online_learning_discrete_sequence_anomalies.py
# ==============================================================================
from brainblocks.blocks import DiscreteTransformer, SequenceLearner
from sklearn import preprocessing

values = [
    'a', 'a', 'a', 'a', 'a', 'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a', 'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a', 'b', 'c', 'g', 'e', 'f']

scores = [0.0 for _ in range(len(values))]

# Convert values to integers
le = preprocessing.LabelEncoder()
le.fit(values)
integers = le.transform(values)

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

# Loop through the values
for i in range(len(integers)):

    # Set scalar transformer value
    lt.set_value(integers[i])

    # Compute the scalar transformer
    lt.feedforward()

    # Compute the sequence learner
    sl.feedforward(learn=True)

    # Get anomaly score
    scores[i] = sl.get_anomaly_score()

# Print output
print("val, scr")
for i in range(len(values)):
    print("%3s, %0.1f" % (values[i], scores[i]))
