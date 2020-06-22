from brainblocks.blocks import ScalarEncoder, PatternClassifier
from sklearn import preprocessing

# define train and test data (x) and labels (y)
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

# string symbols converted to integers
le = preprocessing.LabelEncoder()
le.fit(y_trains)
y_trains_ints = le.transform(y_trains)

# retrieve the integer classes from above
int_classes = [k for k in range(len(le.classes_))]

# define blocks
se = ScalarEncoder(
    min_val=-1.0, # minimum input value
    max_val=1.0,  # maximum input value
    num_s=1024,   # number of statelets
    num_as=128)   # number of active statelets

pp = PatternClassifier(
    labels=int_classes, # user-defined labels
    num_s=512,          # number of statelets
    num_as=8,           # number of active statelets
    perm_thr=20,        # receptor permanence threshold
    perm_inc=2,         # receptor permanence increment
    perm_dec=1,         # receptor permanence decrement
    pct_pool=0.8,       # pooling percentage
    pct_conn=0.5,       # initially connected percentage
    pct_learn=0.25)     # learn percentage

# connect blocks
pp.input.add_child(se.output)

# fit
for i in range(len(x_trains)):
    se.compute(x_trains[i])
    pp.compute(y_trains_ints[i], learn=True)

# predict
probs = []
for i in range(len(x_tests)):
    se.compute(x_tests[i])
    pp.compute(0, learn=True)
    probs.append(pp.get_probabilities())

# print output
print("x,   p_a, p_b")
for i in range(len(x_tests)):
    print("%0.1f, %0.1f, %0.1f" % (x_tests[i], probs[i][0], probs[i][1]))