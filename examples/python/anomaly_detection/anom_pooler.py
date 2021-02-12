# ==============================================================================
# anom_pooler.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, PatternPooler, SequenceLearner

values = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.2, 1.0, 1.0] # <-- anomaly is 0.2

scores = [0.0 for _ in range(len(values))]

# Setup blocks
st = ScalarTransformer(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=1000,  # number of statelets
    num_as=100)  # number of active statelets

pp = PatternPooler(
    num_s=500,     # number of statelets
    num_as=8,      # number of active statelets
    perm_thr=20,   # receptor permanence threshold
    perm_inc=2,    # receptor permanence increment
    perm_dec=1,    # receptor permanence decrement
    pct_pool=0.8,  # pooling percentage
    pct_conn=0.5,  # initially connected percentage
    pct_learn=0.3) # learn percentage

sl = SequenceLearner(
    num_spc=10,  # number of statelets per column
    num_dps=10,  # number of dendrites per statelet
    num_rpd=12,  # number of receptors per dendrite
    d_thresh=6,  # dendrite threshold
    perm_thr=20, # receptor permanence threshold
    perm_inc=2,  # receptor permanence increment
    perm_dec=1)  # receptor permanence decrement

# Connect blocks
pp.input.add_child(st.output, 0)
sl.input.add_child(pp.output, 0)

# Loop through the values
for i in range(len(values)):

    # Set scalar transformer value
    st.set_value(values[i])

    # Compute the scalar transformer
    st.feedforward()

    # Compute the pattern pooler
    pp.feedforward(learn=True)

    # Compute the sequence learner
    sl.feedforward(learn=True)

    # Get anomaly score
    scores[i] = sl.get_anomaly_score()

# Print output
print("val, scr")
for i in range(len(scores)):
    print("%0.1f, %0.1f" % (values[i], scores[i]))
