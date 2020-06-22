from brainblocks.blocks import SymbolsEncoder
from sklearn import preprocessing
import numpy as np

# printing boolean arrays neatly
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=100,
                    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

# list of labels
LABELS = ['a', 'b', 'c']

# use scikit-learn to convert string labels to discrete integers
le = preprocessing.LabelEncoder()
le.fit(LABELS)
int_labels = le.transform(LABELS)

# create the encoder
se_0 = SymbolsEncoder(max_symbols=4, num_s=256)

# convert scalars to distributed binary representation
for i in range(len(int_labels)):
    # encode scalars
    se_0.compute(int_labels[i])

    # list of 0s and 1s representing the distributed binary representation
    intarray = se_0.output.bits

    # converted to numpy array for visualization
    binary_array = np.array(intarray, dtype=np.bool)

    print(int_labels[i])
    print(binary_array)
