# ==============================================================================
# templates/classifier.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, PatternClassifier

class Classifier():
    def __init__(
            self,
            configs=(),     # block configuration
            num_l=2,        # number of labels
            min_val=0.0,    # minimum input value
            max_val=1.0,    # maximum input value
            num_i=1024,     # number of input statelets
            num_ai=128,     # number of active input statelets
            num_s=512,      # number of statelets
            num_as=8,       # number of active statelets
            pct_pool=0.8,   # pooling percentage
            pct_conn=0.5,   # initially connected percentage
            pct_learn=0.3): # learn percentage

        PERM_THR = 20
        PERM_INC = 2
        PERM_DEC = 1

        # seed the random number generator
        # bb.seed(0) # TODO: fix seeding

        self.st = ScalarTransformer(min_val, max_val, num_i, num_ai)

        self.pc = PatternClassifier(num_l, num_s, num_as, PERM_THR, PERM_INC,
                                    PERM_DEC, pct_pool, pct_conn, pct_learn)

        self.pc.input.add_child(self.st.output, 0)

        self.pc.init()

    #def save(self, path='./', name='classifier'):
    #    self.pc.save(path + name + "_pc.bin")

    #def load(self, path='./', name='classifier'):
    #    self.pc.load(path + name + "_pc.bin")

    def fit(self, value=0.0, label=0):

        self.st.set_value(value)
        self.pc.set_label(label)
        self.st.feedforward()
        self.pc.feedforward(learn=True)

        return self.pc.get_probabilities()

    def predict(self, value=0.0):

        self.st.set_value(value)
        self.st.feedforward()
        self.pc.feedforward(learn=False)

        return self.pc.get_probabilities()
