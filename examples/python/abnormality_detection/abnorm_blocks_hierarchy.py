from brainblocks.blocks import ScalarEncoder, PatternPooler, SequenceLearner

# define data and scores
num_inputs = 20
inputs_0 = [0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8]
inputs_1 = [0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0]
scores_0 = [0 for i in range(num_inputs)]
scores_1 = [0 for i in range(num_inputs)]
scores_2 = [0 for i in range(num_inputs)]

# define blocks
se_0 = ScalarEncoder(num_s=500, num_as=50)
se_1 = ScalarEncoder(num_s=500, num_as=50)
pp_0 = PatternPooler(num_s=250, num_as=8)
pp_1 = PatternPooler(num_s=250, num_as=8)
pp_2 = PatternPooler(num_s=250, num_as=8)
sl_0 = SequenceLearner()
sl_1 = SequenceLearner()
sl_2 = SequenceLearner()

# connect blocks
pp_0.input.add_child(se_0.output)
pp_1.input.add_child(se_1.output)
pp_2.input.add_child(pp_0.output)
pp_2.input.add_child(pp_1.output)
sl_0.input.add_child(pp_0.output)
sl_1.input.add_child(pp_1.output)
sl_2.input.add_child(pp_2.output)

# loop through data
for i in range(num_inputs):

    # compute
    se_0.compute(inputs_0[i])
    se_1.compute(inputs_1[i])
    pp_0.compute()
    pp_1.compute()
    pp_2.compute()
    sl_0.compute()
    sl_1.compute()
    sl_2.compute()

    # get scores
    scores_0[i] = sl_0.get_score()
    scores_1[i] = sl_1.get_score()
    scores_2[i] = sl_2.get_score()

# print output
print("in0, in1, sc0, sc1, sc2")
for i in range(num_inputs):
    print("%0.1f, %0.1f, %0.1f, %0.1f, %0.1f" % (inputs_0[i], inputs_1[i], scores_0[i], scores_1[i], scores_2[i]))
