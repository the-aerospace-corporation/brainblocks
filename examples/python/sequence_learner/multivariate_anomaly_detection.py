# ==============================================================================
# multivariate_anomaly_detection.py
# ==============================================================================
from brainblocks.blocks import BlankBlock, SequenceLearner
from brainblocks.tools import HyperGridTransform
from brainblocks.datasets.time_series import make_sample_times, generate_multi_square, generate_sine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_name = "multivariate_output"

# GENERATE TIME-SERIES DATA
num_params = 4

# sample times, 1s and 100Hz
secs = 1
sample_rate = 100

# nominal range
sample_times = make_sample_times(secs=secs, sample_rate=sample_rate)

# abnormal range
abnormal_times = make_sample_times(secs=secs, sample_rate=sample_rate) + secs

param_values = []
param_names = []

param_dict = {}

for k in range(num_params):
    # generate nominal and abnormal portion
    index0, values0 = generate_sine(sample_times, has_spatial_abnormality=False)
    index1, values1 = generate_sine(abnormal_times, has_spatial_abnormality=True)

    # stitch the sequence together
    values = np.concatenate((values0, values1))

    param_dict["value_%d" % k] = values

# stitch the time values together
index = pd.Index(np.concatenate((sample_times, abnormal_times)), name="time")

# put data into pandas dataframe
sigDF = pd.DataFrame(index=index, data=param_dict)
X = sigDF.to_numpy()

# num_bits = num_grids * (num_bins ** num_subspace_dims)
# FIXME: make sure num_bits is a factor of 32
# create hypergrid transform for encoding data
hgt = HyperGridTransform(num_grids=8, num_bins=8, num_subspace_dims=1)

# fit the data
hgt.fit(X)

# transform scalar feature vectors to distributed binary representation
X_bits = hgt.transform(X)

# BLOCKS
# NOTE: num_bits of BlankBlock and num_c of SequenceLearner must be equal!!

# Blank Block to hold the hypergrid output
b0 = BlankBlock(num_s=hgt.num_bits)

# Sequence learner of distributed binary representations
sl = SequenceLearner(
    num_spc=10, # number of statelets per column
    num_dps=10, # number of coincidence detectors per statelet
    num_rpd=12, # number of receptors per coincidence detector
    d_thresh=6, # coincidence detector threshold
    perm_thr=1, # receptor permanence threshold
    perm_inc=1, # receptor permanence increment
    perm_dec=0) # receptor permanence decrement

# connect blank block containing hypergrid data to sequence learner
sl.input.add_child(b0.output)

scores = []
for k in range(len(X_bits)):

    # already converted data, flatten row to 1D array
    X_array = X_bits[k, :].flatten()

    # put it into a blank block
    b0.output.bits = X_array
    b0.feedforward()

    # learn the sequence
    sl.feedforward(learn=True)

    # get the abnormality score
    score = sl.get_anomaly_score()

    scores.append(score)

sigDF["score"] = scores

# save to CSV file
print("Saving to " + output_name + ".csv")
sigDF.to_csv(output_name + ".csv", index=True)

# generate plot of data
print("Saving to " + output_name + ".png")
axes = sigDF.plot(subplots=True, legend=False)
for k in range(len(sigDF.columns)):
    axes[k].set_ylabel(sigDF.columns[k])
plt.savefig(output_name + ".png")
plt.close()
