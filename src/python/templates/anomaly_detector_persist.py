# ==============================================================================
# templates/anomaly_detector_persist.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, PersistenceTransformer, PatternPooler, SequenceLearner

class AnomalyDetectorPersist():
    def __init__(
            self,
            min_val=0.0,    # minimum input value
            max_val=1.0,    # maximum input value
            max_step=8,     # maximum persistence step
            num_i=1024,     # number of input statelets
            num_ai=128,     # number of active input statelets
            num_s=512,      # number of statelets
            num_as=8,       # number of active statelets
            num_spc=10,     # number of statelets per column
            num_dps=10,     # number of dendrites per statelet
            num_rpd=12,     # number of receptors per dendrite
            d_thresh=6,     # dendrite threshold
            pct_pool=0.8,   # pooling percentage
            pct_conn=0.5,   # initially connected percentage
            pct_learn=0.3): # learn percentage

        PERM_THR = 20
        PERM_INC = 2
        PERM_DEC = 1

        num_i_half = int(num_i / 2)
        num_ai_half = int(num_ai / 2)

        self.min_val = min_val
        self.max_val = max_val

        # seed the random number generator
        #bb.seed(0) # TODO: fix seeding

        self.st = ScalarTransformer(min_val, max_val, num_i_half, num_ai_half)

        self.pt = PersistenceTransformer(min_val, max_val, num_i_half,
                                         num_ai_half, max_step)

        self.pp = PatternPooler(num_s, num_as, PERM_THR, PERM_INC, PERM_DEC,
                                pct_pool, pct_conn, pct_learn)

        self.sl = SequenceLearner(num_s, num_spc, num_dps, num_rpd, d_thresh,
                                  PERM_THR, PERM_INC, PERM_DEC)

        self.pp.input.add_child(self.st.output, 0)
        self.pp.input.add_child(self.pt.output, 0)
        self.sl.input.add_child(self.pp.output, 0)

        self.pp.init()
        self.sl.init()

    #def save(self, path='./', name='detector'):
    #    self.pp.save(path + name + "_pp.bin")
    #    self.sl.save(path + name + "_sl.bin")

    #def load(self, path='./', name='detector'):
    #    self.pp.load(path + name + "_pp.bin")
    #    self.sl.load(path + name + "_sl.bin")

    def feedforward(self, value=0.0, learn=True):

        in_bounds = True

        if value < self.min_val or value > self.max_val:
            in_bounds = False

        self.st.set_value(value)
        self.pt.set_value(value)
        self.st.feedforward()
        self.pt.feedforward()
        self.pp.feedforward(learn)
        self.sl.feedforward(learn)

        #print(self.pp.input.acts)
        #print(self.pt.output.acts)
        #print()

        if in_bounds:
            anom = self.sl.get_anomaly_score()
        else:
            anom = 1.0

        return anom
