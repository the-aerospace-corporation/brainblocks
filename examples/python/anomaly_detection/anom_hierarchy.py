# ==============================================================================
# anom_hierarchy.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, PatternPooler, SequenceLearner

num_values = 20

values0 = [
    0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8,
    0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8]

values1 = [
    0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0,
    0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0]

scores0 = [0 for i in range(num_values)]
scores1 = [0 for i in range(num_values)]
scores2 = [0 for i in range(num_values)]

# Setup blocks
st0 = ScalarTransformer(num_s=500, num_as=50)
st1 = ScalarTransformer(num_s=500, num_as=50)
pp0 = PatternPooler(num_s=250, num_as=8)
pp1 = PatternPooler(num_s=250, num_as=8)
pp2 = PatternPooler(num_s=250, num_as=8)
sl0 = SequenceLearner()
sl1 = SequenceLearner()
sl2 = SequenceLearner()

# Connect blocks
pp0.input.add_child(st0.output, 0)
pp1.input.add_child(st1.output, 0)
pp2.input.add_child(pp0.output, 0)
pp2.input.add_child(pp1.output, 0)
sl0.input.add_child(pp0.output, 0)
sl1.input.add_child(pp1.output, 0)
sl2.input.add_child(pp2.output, 0)

# Loop through the values
for i in range(num_values):

    # Compute hierarchy
    st0.set_value(values0[i])
    st1.set_value(values1[i])
    st0.feedforward()
    st1.feedforward()
    pp0.feedforward(learn=True)
    pp1.feedforward(learn=True)
    pp2.feedforward(learn=True)
    sl0.feedforward(learn=True)
    sl1.feedforward(learn=True)
    sl2.feedforward(learn=True)

    # Get scores
    scores0[i] = sl0.get_anomaly_score()
    scores1[i] = sl1.get_anomaly_score()
    scores2[i] = sl2.get_anomaly_score()

# Print output
print("in0, in1, sl0, sl1, sl2")
for i in range(num_values):
    print("%0.1f, %0.1f, %0.1f, %0.1f, %0.1f" % (
        values0[i], values1[i], scores0[i], scores1[i], scores2[i]))
