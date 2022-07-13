# ==============================================================================
# online_learning_scalar_sequence_anomalies.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, SequenceLearner

train_values = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

test_values = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.2, 1.0, 1.0] # <-- anomaly is 0.2

train_scores = [0.0 for _ in range(len(train_values))]
test_scores = [0.0 for _ in range(len(test_values))]

# Setup blocks
st = ScalarTransformer(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=64,    # number of statelets
    num_as=8)    # number of active statelets

sl = SequenceLearner(
    num_spc=10,  # number of statelets per column
    num_dps=10,  # number of dendrites per statelet
    num_rpd=12,  # number of receptors per dendrite
    d_thresh=6,  # dendrite threshold
    perm_thr=20, # receptor permanence threshold
    perm_inc=2,  # receptor permanence increment
    perm_dec=1)  # receptor permanence decrement

# Connect blocks
sl.input.add_child(st.output, 0)

# Loop through the values twice
for iteration_i in range(2):
    for i in range(len(train_values)):

        # Set scalar transformer value
        st.set_value(train_values[i])

        # Compute the scalar transformer
        st.feedforward()

        # Compute the sequence learner w/ learning
        sl.feedforward(learn=True)

        # Get anomaly score
        train_scores[i] = sl.get_anomaly_score()

# Reset SequenceLearner
st.clear()
sl.clear()

# Loop through the values
for i in range(len(test_values)):

    # Set scalar transformer value
    st.set_value(test_values[i])

    # Compute the scalar transformer
    st.feedforward()

    # Compute the sequence learner w/o learning
    sl.feedforward(learn=False)

    # Get anomaly score
    test_scores[i] = sl.get_anomaly_score()

# Print output
print("TRAIN")
print("val, scr")
for i in range(len(train_values)):
    print("%0.1f, %0.1f" % (train_values[i], train_scores[i]))

# Print output
print()
print("TEST")
print("val, scr")
for i in range(len(test_values)):
    print("%0.1f, %0.1f" % (test_values[i], test_scores[i]))

