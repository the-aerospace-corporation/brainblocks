from brainblocks.templates import Classifier
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

c = Classifier(
    labels=int_classes, # PatternClassifier labels
    min_val=0.0,        # ScalarEncoder minimum input value
    max_val=1.0,        # ScalarEncoder maximum input value
    num_i=1024,         # ScalarEncoder number of statelets
    num_ai=128,         # ScalarEncoder number of active statelets
    num_s=512,          # PatternClassifier number of statelets
    num_as=8,           # PatternClassifier number of active statelets
    pct_pool=0.8,       # PatternClassifier pool percentage
    pct_conn=0.5,       # PatternClassifier initial connection percentage
    pct_learn=0.25)     # PatternClassifier learn percentage

# fit
c.fit([x_trains], y_trains_ints)

# predict
probs = c.predict([x_tests])

# print output
print("x,   p_a, p_b")
for i in range(len(x_tests)):
    print("%0.1f, %0.1f, %0.1f" % (x_tests[i], probs[i][0], probs[i][1]))
