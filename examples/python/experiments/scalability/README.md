# Benchmarks

## Scalability

- **Hardware**: Intel(R) Core(TM) i5-6600K CPU @ 3.50GHz, 3501 Mhz, 4 Core(s), 4 Logical Processor(s)

### Varying Number of Anomaly Detectors

Adding anomaly detectors should have a linear effect on memory usage, initialization time, and average compute time per step.

Single thread as of 2021-01-04:

```
$ python anomaly_detector_scalability.py
+-------------------------+-----------------------------+
| Performance Test        | CPU Freq: 3.50 GHz          |
+------------+------------+-------------+---------------+
| Detectors  | Mem Usage  |  Init Time  | Avg Comp Time |
+------------+------------+-------------+---------------+
|          1 |   54.09 MB |  0.013962 s |    0.000310 s |
|          2 |   63.06 MB |  0.027925 s |    0.000585 s |
|          4 |   73.70 MB |  0.056848 s |    0.001238 s |
|          8 |   94.94 MB |  0.112727 s |    0.002476 s |
|         16 |  137.56 MB |  0.222508 s |    0.004986 s |
|         32 |  222.68 MB |  0.442945 s |    0.009815 s |
|         64 |  392.59 MB |  0.897606 s |    0.019607 s |
|        128 |  732.75 MB |  1.786823 s |    0.039004 s |
|        256 | 1412.64 MB |  3.575544 s |    0.078669 s |
+------------+------------+-------------+---------------+
```

### Varying Anomaly Detector Size

Increasing the number of dendrites in a single anomaly detector should have a linear effect on memory usage, initialization time, and average compute time per step.

Single thread as of 2021-01-04:

```
$ python dendrite_scalability.py
+-------------------------+-----------------------------+
| Performance Test        | CPU Freq: 3.50 GHz          |
+------------+------------+-------------+---------------+
| Dendrites  | Mem Usage  |  Init Time  | Avg Comp Time |
+------------+------------+-------------+---------------+
|      12928 |   50.07 MB |  0.003989 s |    0.000275 s |
|      25856 |   55.18 MB |  0.006980 s |    0.000275 s |
|      51712 |   57.84 MB |  0.012965 s |    0.000310 s |
|     103424 |   63.09 MB |  0.026950 s |    0.000378 s |
|     206848 |   73.67 MB |  0.056848 s |    0.000446 s |
|     413696 |   94.95 MB |  0.111741 s |    0.000584 s |
|     827392 |  137.51 MB |  0.223416 s |    0.000927 s |
|    1654784 |  222.19 MB |  0.447966 s |    0.001616 s |
|    3309568 |  391.84 MB |  0.895879 s |    0.002993 s |
+------------+------------+-------------+---------------+
```