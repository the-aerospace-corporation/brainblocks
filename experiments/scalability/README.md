# Benchmarks

## Scalability

- **Hardware**: Intel(R) Core(TM) i5-6600K CPU @ 3.50GHz, 3501 Mhz, 4 Core(s), 4 Logical Processor(s)

### Varying Number of Abnormality Detectors

Adding abnormality detectors should have a linear effect on memory usage, initialization time, and average compute time per step.

```
$ python abnormality_detector_scalability.py
+-------------------------+-----------------------------+
| Performance Test        | CPU Freq: 3.50 GHz          |
+------------+------------+-------------+---------------+
| Detectors  | Mem Usage  |  Init Time  | Avg Comp Time |
+------------+------------+-------------+---------------+
|          1 |   58.40 MB |  0.019974 s |    0.000619 s |
|          2 |   71.84 MB |  0.038896 s |    0.001204 s |
|          4 |   91.75 MB |  0.077772 s |    0.002334 s |
|          8 |  130.95 MB |  0.140612 s |    0.004847 s |
|         16 |  209.82 MB |  0.316095 s |    0.009973 s |
|         32 |  367.01 MB |  0.609231 s |    0.019465 s |
|         64 |  681.57 MB |  1.235137 s |    0.037745 s |
|        128 | 1310.75 MB |  2.414196 s |    0.074953 s |
|        256 | 2569.14 MB |  4.924181 s |    0.150354 s |
+------------+------------+-------------+---------------+
```

### Varying Abnormality Detector Size

Increasing the number of coincidence detectors in a single abnormality detector should have a linear effect on memory usage, initialization time, and average compute time per step.

```
$ python coincidence_detector_scalability.py
+-------------------------+-----------------------------+
| Performance Test        | CPU Freq: 3.50 GHz          |
+------------+------------+-------------+---------------+
| Detectors  | Mem Usage  |  Init Time  | Avg Comp Time |
+------------+------------+-------------+---------------+
|      12928 |   50.62 MB |  0.004994 s |    0.000279 s |
|      25856 |   56.81 MB |  0.015621 s |    0.000000 s |
|      51712 |   61.94 MB |  0.015641 s |    0.000539 s |
|     103424 |   71.53 MB |  0.046864 s |    0.000539 s |
|     206848 |   91.42 MB |  0.078128 s |    0.001616 s |
|     413696 |  130.62 MB |  0.156234 s |    0.002693 s |
|     827392 |  209.25 MB |  0.300112 s |    0.005839 s |
|    1654784 |  366.38 MB |  0.630482 s |    0.012235 s |
|    3309568 |  679.34 MB |  1.203679 s |    0.026425 s |
+------------+------------+-------------+---------------+
```