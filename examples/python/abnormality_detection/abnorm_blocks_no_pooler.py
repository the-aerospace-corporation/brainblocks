from brainblocks.blocks import ScalarEncoder, SequenceLearner

# define data and scores
scores = []
values = [
    0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.2, 0.0, 0.0]  # <-- abnormality is 0.2

# define blocks
e = ScalarEncoder(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=64,    # number of statelets
    num_as=8)    # number of active statelets

sl = SequenceLearner(
    num_spc=10, # number of statelets per column
    num_dps=10, # number of coincidence detectors per statelet
    num_rpd=12, # number of receptors per coincidence detector
    d_thresh=6, # coincidence detector threshold
    perm_thr=1, # receptor permanence threshold
    perm_inc=1, # receptor permanence increment
    perm_dec=0) # receptor permanence decrement

# connect block pages
sl.input.add_child(e.output)

# loop through data
for i in range(len(values)):

    # convert data using scalar encoder
    e.compute(values[i])

    # learn the sequence
    sl.compute(learn=True)

    # get the abnormality score
    score = sl.get_score()
    scores.append(score)

# print output
print("value, score")
for i in range(len(scores)):
    print("%0.1f, %0.1f" % (values[i], scores[i]))