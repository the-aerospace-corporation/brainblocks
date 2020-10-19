from brainblocks.blocks import ScalarEncoder, PatternPooler, SequenceLearner

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
    num_s=1000,  # number of statelets
    num_as=100)  # number of active statelets

pp = PatternPooler(
    num_s=500,      # number of statelets
    num_as=8,       # number of active statelets
    perm_thr=20,    # receptor permanence threshold
    perm_inc=2,     # receptor permanence increment
    perm_dec=1,     # receptor permanence decrement
    pct_pool=0.8,   # pooling percentage
    pct_conn=0.5,   # initially connected percentage
    pct_learn=0.25) # learn percentage

sl = SequenceLearner(
    num_spc=10, # number of statelets per column
    num_dps=10, # number of coincidence detectors per statelet
    num_rpd=12, # number of receptors per coincidence detector
    d_thresh=6, # coincidence detector threshold
    perm_thr=1, # receptor permanence threshold
    perm_inc=1, # receptor permanence increment
    perm_dec=0) # receptor permanence decrement

# connect blocks
pp.input.add_child(e.output)
sl.input.add_child(pp.output)

# loop through data
for i in range(len(values)):

    # convert data using scalar encoder
    e.compute(values[i])

    # pool the pattern
    pp.compute(learn=True)

    # learn the sequence
    sl.compute(learn=True)

    # get the abnormality score
    score = sl.get_score()
    scores.append(score)

# print output
print("value, score")
for i in range(len(scores)):
    print("%0.1f, %0.1f" % (values[i], scores[i]))