import os
import platform
import psutil
import time
import numpy as np
import matplotlib.pyplot as plt
from brainblocks.blocks import ScalarTransformer, PatternPooler, SequenceLearner

data = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0]

expected_scores = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0])

def get_scalability(num_detectors):
    process = psutil.Process(os.getpid())
    scores = []

    # Setup Blocks
    transformers = []
    pattern_poolers = []
    sequence_learners = []
    for _ in range(num_detectors):
        transformers.append(ScalarTransformer(min_val=0.0, max_val=1.0, num_s=1024, num_as=128))
        pattern_poolers.append(PatternPooler(num_s=512, num_as=8, pct_pool=0.8, pct_conn=0.5, pct_learn=0.3))
        sequence_learners.append(SequenceLearner(num_c=512, num_spc=10, num_dps=10, num_rpd=12, d_thresh=6))
        pattern_poolers[-1].input.add_child(transformers[-1].output, 0)
        sequence_learners[-1].input.add_child(pattern_poolers[-1].output, 0)

    # Get initialization time and memory usage
    t0 = time.time()
    for d in range(num_detectors):
        pattern_poolers[d].init()
        sequence_learners[d].init()
    t1 = time.time()
    init_time = t1 - t0
    num_bytes = process.memory_info().rss

    # Get compute time
    t0 = time.time()
    for d in range(num_detectors):
        for i in range(len(data)):
            transformers[d].set_value(data[i])
            transformers[d].feedforward()
            pattern_poolers[d].feedforward(learn=True)
            sequence_learners[d].feedforward(learn=True)
            if (d == 0):
                score = sequence_learners[d].get_anomaly_score()
                scores.append(score)
    t1 = time.time()
    comp_time = t1 - t0

    # Test Results
    np.testing.assert_array_equal(np.array(scores), expected_scores)

    return [num_detectors, num_bytes, init_time, comp_time]

if __name__ == "__main__":
    num_detectors = [] # number of abnormality detectors
    num_megabytes = []
    init_times = []
    avg_comp_times = []

    tests = [None] * 9
    tests[0] = get_scalability(1)
    tests[1] = get_scalability(2)
    tests[2] = get_scalability(4)
    tests[3] = get_scalability(8)
    tests[4] = get_scalability(16)
    tests[5] = get_scalability(32)
    tests[6] = get_scalability(64)
    tests[7] = get_scalability(128)
    tests[8] = get_scalability(256)

    # Print Results
    print("+-------------------------+-----------------------------+")
    print("| Performance Test        | CPU Freq: {:0.2f} GHz          | ".format(psutil.cpu_freq().current / 1000))
    print("+------------+------------+-------------+---------------+")
    print("| Detectors  | Mem Usage  |  Init Time  | Avg Comp Time |")
    print("+------------+------------+-------------+---------------+")
    for test in tests:
        num_detectors.append(test[0])
        num_megabytes.append(test[1]/1000000)
        init_times.append(test[2])
        avg_comp_times.append(test[3]/(len(data)-1)) # skipped first data step because blocks are initialized there
        print("| {:10d} | {:7.2f} MB | {:9.6f} s | {:11.6f} s |".format(num_detectors[-1], num_megabytes[-1], init_times[-1], avg_comp_times[-1]))
    print("+------------+------------+-------------+---------------+")

    # Plot Results
    fig, ax = plt.subplots(nrows=3, sharex=True)
    fig.suptitle("Abnormality Detector Scalability", fontsize=12)
    ax[0].set_ylabel("Memory Usage (MB)", fontsize = 8)
    ax[1].set_ylabel("Init Time (s)", fontsize = 8)
    ax[2].set_ylabel("Comp Time per Step (s)", fontsize = 8)
    ax[2].set_xlabel("Abnormality Detectors", fontsize = 10)
    ax[0].plot(num_detectors, num_megabytes, color="r", marker="o")
    ax[1].plot(num_detectors, init_times, color="g", marker="o")
    ax[2].plot(num_detectors, avg_comp_times, color="b", marker="o")
    plt.plot()
    plt.savefig("abnormality_detector_scalability.png")
    plt.close()

    # Plot Benchmark Input
    x = [i for i in range(len(data))]
    fig, ax = plt.subplots(nrows=2, sharex=True)
    fig.suptitle("Benchmark Input", fontsize=12)
    ax[0].set_ylabel("Data", fontsize = 10)
    ax[1].set_ylabel("Expected Anom", fontsize = 10)
    ax[1].set_xlabel("Step", fontsize = 10)
    ax[0].plot(x, data, color="b", marker="o")
    ax[1].plot(x, expected_scores, color="r", marker="o")
    plt.plot()
    plt.savefig("benchmark_input.png")
    plt.close()