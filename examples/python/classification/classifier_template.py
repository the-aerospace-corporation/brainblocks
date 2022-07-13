# ==============================================================================
# classifier_template.py
# ==============================================================================
from brainblocks.templates import Classifier
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

# Setup classifier
c = Classifier(
    num_l=2,       # PatternClassifier number of labels
    min_val=0.0,   # ScalarEncoder minimum input value
    max_val=1.0,   # ScalarEncoder maximum input value
    num_i=1024,    # ScalarEncoder number of statelets
    num_ai=128,    # ScalarEncoder number of active statelets
    num_s=512,     # PatternClassifier number of statelets
    num_as=8,      # PatternClassifier number of active statelets
    pct_pool=0.8,  # PatternClassifier pool percentage
    pct_conn=0.5,  # PatternClassifier initial connection percentage
    pct_learn=0.3) # PatternClassifier learn percentage

# Fit
for i in range(len(x_trains)):
    c.fit(x_trains[i], y_trains_ints[i])

# Predict
probs = [[], []]
for j in range(len(x_tests)):
    probs[j] = c.predict(x_tests[j])

# Print output
print("x,   p_a, p_b")
for j in range(len(x_tests)):
    print("%0.1f, %0.1f, %0.1f" % (x_tests[j], probs[j][0], probs[j][1]))
