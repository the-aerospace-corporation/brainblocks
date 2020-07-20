from brainblocks.blocks import SymbolsEncoder, SequenceLearner
from sklearn import preprocessing

# Example showing the use of reset function for sequence learning
# Useful when learning batches of finite sequences instead of online learning of streaming data

# NOTE:  the first data point after reset is always abnormal since no predictions are made
# input is always unexpected and triggers a surprise update

# define data and scores
scores = []
values = [
    'a', 'a', 'a', 'a', 'a',
    'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a',
    'a', 'a', 'a', 'a', 'a',
    'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a',
    'b', 'c', 'd', 'e', 'f',
    'a', 'a', 'a', 'a', 'a']

# convert to integer values
le = preprocessing.LabelEncoder()
le.fit(values)
int_values = le.transform(values)

# define blocks
e = SymbolsEncoder(
    max_symbols=26,  # maximum number of symbols
    num_s=208)  # number of statelets

sl = SequenceLearner(
    num_spc=10,  # number of statelets per column
    num_dps=10,  # number of coincidence detectors per statelet
    num_rpd=12,  # number of receptors per coincidence detector
    d_thresh=6,  # coincidence detector threshold
    perm_thr=1,  # receptor permanence threshold
    perm_inc=1,  # receptor permanence increment
    perm_dec=0)  # receptor permanence decrement

# connect block pages
sl.input.add_child(e.output)

# repeat 3 times
for iteration_i in range(3):

    # loop through data
    for i in range(len(int_values)):
        # compute
        e.compute(int_values[i])
        sl.compute(learn=True)

        # get the abnormality score
        score = sl.get_score()
        scores.append(score)

    # print output
    print("iteration %d" % iteration_i)
    print("value, score")
    for i in range(len(scores)):
        print("%5s, %0.1f" % (values[i], scores[i]))
    print()
    scores = []

    # reset sequence by clearing the blocks
    e.clear()
    sl.clear()
