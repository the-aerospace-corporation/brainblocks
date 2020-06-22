from brainblocks.blocks import ScalarEncoder
import numpy as np

# printing boolean arrays neatly
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=100,
                    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

values = [
    0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0]

se_0 = ScalarEncoder(num_s=1024, num_as=128)

# convert scalars to distributed binary representation
for i in range(len(values)):

    # encode scalars
    se_0.compute(values[i])

    # list of 0s and 1s representing the distributed binary representation
    intarray = se_0.output.bits

    # converted to numpy array for visualization
    binary_array = np.array(intarray, dtype=np.bool)

    print(values[i])
    print(binary_array)
