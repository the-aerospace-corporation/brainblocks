# ==============================================================================
# pattern_classifier.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, PatternClassifier
from sklearn import preprocessing

x_trains = [
    0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0]

y_trains = [
    "a", "b", "a", "b", "a",
    "b", "a", "b", "a", "b",
    "a", "b", "a", "b", "a",
    "b", "a", "b", "a", "b",
    "a", "b", "a", "b", "a",
    "b", "a", "b", "a", "b",
    "a", "b", "a", "b", "a"]

x_tests = [0.0, 1.0]

# Convert to integer labels
le = preprocessing.LabelEncoder()
le.fit(y_trains)
y_trains_ints = le.transform(y_trains)

# Setup blocks
st = ScalarTransformer(
    min_val=-1.0, # minimum input value
    max_val=1.0,  # maximum input value
    num_s=1024,   # number of statelets
    num_as=128)   # number of active statelets

pp = PatternClassifier(
    num_l=2,       # number of labels
    num_s=512,     # number of statelets
    num_as=8,      # number of active statelets
    perm_thr=20,   # receptor permanence threshold
    perm_inc=2,    # receptor permanence increment
    perm_dec=1,    # receptor permanence decrement
    pct_pool=0.8,  # percent pooled
    pct_conn=0.5,  # percent initially connected
    pct_learn=0.3) # percent learn

# Connect blocks
pp.input.add_child(st.output, 0)

# Fit
for i in range(len(x_trains)):
    st.set_value(x_trains[i])
    pp.set_label(y_trains_ints[i])
    st.feedforward()
    pp.feedforward(learn=True)

# Predict
probs = []
for i in range(len(x_tests)):
    st.set_value(x_tests[i])
    st.feedforward()
    pp.feedforward(learn=True)
    probs.append(pp.get_probabilities())

# Print output
print("x,   p_a, p_b")
for i in range(len(x_tests)):
    print("%0.1f, %0.1f, %0.1f" % (x_tests[i], probs[i][0], probs[i][1]))
