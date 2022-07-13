# SequenceLearner Examples

- Batch Learning Anomaly Detection
    - [train_test_scalar_sequence_anomalies.py](train_test_scalar_sequence_anomalies.py)
        - Use `ScalarTransform` and `SequenceLearner`
    - [train_test_discrete_sequence_anomalies.py](train_test_discrete_sequence_anomalies.py)
        - Use `DiscreteTransform` and `SequenceLearner`

- Online Learning Anomaly Detection
    - [online_learning_scalar_sequence_anomalies.py](online_learning_scalar_sequence_anomalies.py)
        - Use `ScalarTransform` and `SequenceLearner`
    - [online_learning_discrete_sequence_anomalies.py](online_learning_discrete_sequence_anomalies.py)
        - Use `DiscreteTransform` and `SequenceLearner`

- Save and Load Trained Model to File
    - [load_save_sequence_learner.py](load_save_sequence_learner.py)
        - Train, save, load, and compare learned model with `ScalarTransform` and `SequenceLearner`
    
- Template
    - [template_anomaly_detector.py](template_anomaly_detector.py)
        - Example of anomaly detection using the `AnomalyDetector` template

- Pooling
    - [anom_pooler.py](anom_pooler.py)
        - Example of anomaly detection with `PatternPooler` block

- Multivariate
    - [anom_hierarchy.py](anom_hierarchy.py)
        - Example of anomaly detection on multiple time series using a hierarchy of blocks
    - [multivariate_anomaly_detection.py](multivariate_anomaly_detection.py)
        - Example of multivariate time-series anomaly detection
        
