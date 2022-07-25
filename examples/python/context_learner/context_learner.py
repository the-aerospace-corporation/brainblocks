# ==============================================================================
# context_learner.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, ContextLearner
import numpy as np

# Printing boolean arrays neatly
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=80,
    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

# train input & context scalar values
input_values =   [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
context_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# test input & context scalar values
input_values2 =   [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
context_values2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


# input values
st1 = ScalarTransformer(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=128,  # number of statelets
    num_as=16)  # number of active statelets

# context values
st2 = ScalarTransformer(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=128,  # number of statelets
    num_as=16)  # number of active statelets

cl = ContextLearner(
    num_spc=10,  # number of statelets per column
    num_dps=10,  # number of coincidence detectors per statelet
    num_rpd=8,  # number of receptors per coincidence detector
    d_thresh=5,  # coincidence detector threshold
    perm_thr=20,  # receptor permanence threshold
    perm_inc=2,  # receptor permanence increment
    perm_dec=1)  # receptor permanence decrement

# connect encoded input to ContextLearner input
cl.input.add_child(st1.output)

# connect encoded context to ContextLearner context
cl.context.add_child(st2.output)

train_inputs = []
train_contexts = []
train_scores = []

test_inputs = []
test_contexts = []
test_scores = []

# train through the values 2 times
for iteration_i in range(2):
    for i in range(len(input_values)):

        # Set scalar transformer value
        st1.set_value(input_values[i])
        st2.set_value(context_values[i])

        # Compute the scalar transformer
        st1.feedforward()
        st2.feedforward()

        # Compute the ContextLearner
        cl.feedforward(learn=True)

        # store values
        train_inputs.append(input_values[i])
        train_contexts.append(context_values[i])
        train_scores.append(cl.get_anomaly_score())

# test through the values 1 times
for iteration_i in range(1):
    for i in range(len(input_values2)):

        # Set scalar transformer value
        st1.set_value(input_values2[i])
        st2.set_value(context_values2[i])

        # Compute the scalar transformer
        st1.feedforward()
        st2.feedforward()

        # Compute the ContextLearner
        cl.feedforward(learn=True)

        # store values
        test_inputs.append(input_values2[i])
        test_contexts.append(context_values2[i])
        test_scores.append(cl.get_anomaly_score())

print()
print("COLUMNS")
print("input,context,anomaly_score")
print()

print("TRAIN")
for i in range(len(train_scores)):
    print(train_inputs[i], train_contexts[i], train_scores[i])
print()

print("TEST")
for i in range(len(test_scores)):
    print(test_inputs[i], test_contexts[i], test_scores[i])
print()
