# ==============================================================================
# discrete_transformer.py
# ==============================================================================
from brainblocks.blocks import DiscreteTransformer
from sklearn import preprocessing
import numpy as np

# Printing boolean arrays neatly
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=80,
    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

# List of labels
LABELS = ['a', 'b', 'c', 'd']

# Use scikit-learn to convert string labels to discrete integers
le = preprocessing.LabelEncoder()
le.fit(LABELS)
int_labels = le.transform(LABELS)

# create the transformer
lt = DiscreteTransformer(
    num_v=4,   # number of discrete values
    num_s=128) # number of statelets

# Convert scalars to distributed binary representation
for i in range(len(int_labels)):

    # Set label transformer value
    lt.set_value(int_labels[i])

    # Compute label transformer
    lt.feedforward()

    # List of 0s and 1s representing the distributed binary representation
    intarray = lt.output.bits

    # converted to numpy array for visualization
    binary_array = np.array(intarray, dtype=np.bool)

    print(int_labels[i])
    print(binary_array)
