import brainblocks.bb_backend as bb
from brainblocks.blocks import ScalarEncoder, PatternClassifier
from .template_helper import get_blocks

class Classifier():
    def __init__(
            self, 
            configs=(),      # block configuration
            labels=(0,1),    # labels
            min_val=-1.0,    # ScalarEncoder minimum input value
            max_val=1.0,     # ScalarEncoder maximum input value
            num_i=1024,      # ScalarEncoder number of statelets
            num_ai=128,      # ScalarEncoder number of active statelets
            num_s=32,        # PatternClassifier number of statelets
            num_as=8,        # PatternClassifier number of active statelets
            pct_pool=0.8,    # PatternClassifier pool percentage
            pct_conn=0.5,    # PatternClassifier initial connection percentage
            pct_learn=0.25): # PatternClassifier learn percentage

        # seed the random number generator
        bb.seed(0)

        # build blocks from config descriptions if given
        blocks = get_blocks(configs)
        self.encoders = blocks["encoders"]
        self.pc = blocks["pattern_classifier"]

        if len(self.encoders) == 0:
            self.encoders.append(ScalarEncoder(min_val, max_val, num_i, num_ai))

        if self.pc == None:
            num_l = len(labels)
            self.pc = PatternClassifier(labels, num_s, num_as, 20, 2, 1,
                                        pct_pool, pct_conn, pct_learn)

        for encoder in self.encoders:
            self.pc.input.add_child(encoder.output)

    def print_parameters(self):
        for encoder in self.encoders:
            encoder.print_parameters()
        self.pc.print_parameters()

    def save_memories(self, path='./', name='classifier'):
        self.pc.save_memories(path + name + "_pc.bin")

    def load_memories(self, path='./', name='classifier'):
        self.pc.load_memories(path + name + "_pc.bin")

    def fit(self, inputs=(), labels=()):
        probs = []
        num_steps = 0xFFFFFFFF
        num_measurands = len(inputs)
        num_encoders = len(self.encoders)

        if num_measurands != num_encoders:
            print("Warning: compute() num_measurands != num_encoders")
            return probs

        for input in inputs:
            len_input = len(input)
            if len_input < num_steps:
                num_steps = len_input

        for s in range(num_steps):
            for e in range(num_encoders):
                self.encoders[e].compute(inputs[e][s])
            self.pc.compute(labels[s], learn=True)
            probs.append(self.pc.get_probabilities())

        return probs

    def predict(self, inputs=()):
        probs = []
        num_steps = 0xFFFFFFFF
        num_measurands = len(inputs)
        num_encoders = len(self.encoders)

        if num_measurands != num_encoders:
            print("Warning: compute() num_measurands != num_encoders")
            return probs

        for input in inputs:
            len_input = len(input)
            if len_input < num_steps:
                num_steps = len_input

        for s in range(num_steps):
            for e in range(num_encoders):
                self.encoders[e].compute(inputs[e][s])
            self.pc.compute(0, learn=False)
            probs.append(self.pc.get_probabilities())

        return probs