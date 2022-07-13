# ==============================================================================
# persistence_transformer.py
# ==============================================================================
from brainblocks.blocks import PersistenceTransformer
import numpy as np

# Printing boolean arrays neatly
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=80,
    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

values = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

pt = PersistenceTransformer(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=128,  # number of statelets
    num_as=16,  # number of active statelets
    max_step=8)  # maxumum persistence step

# Convert persistences to distributed binary representation
for i in range(len(values)):

    # Set persistence transformer value
    pt.set_value(values[i])

    # Compute persistence transformer
    pt.feedforward()

    # List of 0s and 1s representing the distributed binary representation
    intarray = pt.output.bits

    # Converted to numpy array for visualization
    binary_array = np.array(intarray, dtype=np.bool)

    print(values[i])
    print(binary_array)
