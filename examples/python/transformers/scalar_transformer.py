# ==============================================================================
# scalar_transformer.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer
import numpy as np

# Printing boolean arrays neatly
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=80,
    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

values = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

st = ScalarTransformer(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=128,  # number of statelets
    num_as=16)  # number of active statelets

# Convert scalars to distributed binary representation
for i in range(len(values)):

    # Set scalar transformer value
    st.set_value(values[i])

    # Compute scalar transformer
    st.feedforward()

    # List of 0s and 1s representing the distributed binary representation
    intarray = st.output.bits

    # Converted to numpy array for visualization
    binary_array = np.array(intarray, dtype=np.bool)

    print(values[i])
    print(binary_array)
