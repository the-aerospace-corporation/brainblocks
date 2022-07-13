# ==============================================================================
# template_anomaly_detector.py
# ==============================================================================
from brainblocks.templates import AnomalyDetector

values = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.2, 1.0, 1.0] # <-- anomaly is 0.2

scores = [0.0 for _ in range(len(values))]

ad = AnomalyDetector(
    min_val=0.0,   # minumum value
    max_val=1.0,   # maximum value
    num_i=1000,    # ScalarEncoder number of statelets
    num_ai=100,    # ScalarEncoder number of active statelets
    num_s=500,     # PatternPooler number of statelets
    num_as=8,      # PatternPooler number of active statelets
    num_spc=10,    # SequenceLearner number of statelets per column
    num_dps=10,    # SequenceLearner number of dendrites per statelet
    num_rpd=12,    # SequenceLearner number of receptors per dendrite
    d_thresh=6,    # SequenceLearner coincidence detector threshold
    pct_pool=0.8,  # PatternPooler pool percentage
    pct_conn=0.5,  # PatternPooler initial connection percentage
    pct_learn=0.3) # PatternPooler learn percentage

for i in range(len(values)):
    scores[i] = ad.feedforward(values[i])

print("val, scr")
for i in range(len(scores)):
    print("%0.1f, %0.1f" % (values[i], scores[i]))
