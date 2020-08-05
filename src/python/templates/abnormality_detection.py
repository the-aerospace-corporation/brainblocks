import brainblocks.bb_backend as bb
from brainblocks.blocks import ScalarEncoder, PersistenceEncoder, PatternPooler, SequenceLearner
from .template_helper import get_blocks

class AbnormalityDetector():
    def __init__(
            self,
            configs=(),      # block configuration
            min_val=-1.0,    # minumum value
            max_val=1.0,     # maximum value
            num_i=1024,      # ScalarEncoder number of statelets
            num_ai=128,      # ScalarEncoder number of active statelets
            num_s=512,       # PatternPooler number of statelets
            num_as=8,        # PatternPooler number of active statelets
            num_spc=10,      # SequenceLearner number of statelets per column
            num_dps=10,      # SequenceLearner number of coincidence detectors per statelet
            num_rpd=12,      # SequenceLearner number of receptors per coincidence detector
            d_thresh=6,      # SequenceLearner coincidence detector threshold
            pct_pool=0.8,    # PatternPooler pool percentage
            pct_conn=0.5,    # PatternPooler initial connection percentage
            pct_learn=0.25): # PatternPooler learn percentage

        self.min_val = min_val
        self.max_val = max_val

        # seed the random number generator
        bb.seed(0)

        # build blocks from config descriptions if given
        blocks = get_blocks(configs)
        self.encoders = blocks["encoders"]
        self.pp = blocks["pattern_pooler"]
        self.sl = blocks["sequence_learner"]

        if len(self.encoders) == 0:
            self.encoders.append(ScalarEncoder(min_val, max_val, num_i, num_ai))

        if self.pp == None:
            self.pp = PatternPooler(num_s, num_as, 20, 2, 1, pct_pool, pct_conn, pct_learn)

        if self.sl == None:
            self.sl = SequenceLearner(num_spc, num_dps, num_rpd, d_thresh, 1, 1, 0)

        for encoder in self.encoders:
            self.pp.input.add_child(encoder.output)

        self.sl.input.add_child(self.pp.output)
        
        self.initialized = False

    def print_parameters(self):
        for encoder in self.encoders:
            encoder.print_parameters()
        self.pp.print_parameters()
        self.sl.print_parameters()

    def save_memories(self, path='./', name='detector'):
        self.pp.save_memories(path + name + "_pp.bin")
        self.sl.save_memories(path + name + "_sl.bin")

    def load_memories(self, path='./', name='detector'):
        self.pp.load_memories(path + name + "_pp.bin")
        self.sl.load_memories(path + name + "_sl.bin")

    def compute(self, vectors=(), learn=True):
        anoms = []
        num_steps = 0xFFFFFFFF
        num_measurands = len(vectors)
        num_encoders = len(self.encoders)

        if num_measurands != num_encoders:
            print("Warning: compute() num_measurands != num_encoders")
            return anoms

        for vector in vectors:
            len_vector = len(vector)
            if len_vector < num_steps:
                num_steps = len_vector

        for e in range(num_encoders):
            if isinstance(self.encoders[e], PersistenceEncoder):
                self.encoders[e].reset()

        limit_flag = 0
        for s in range(num_steps):
            for e in range(num_encoders):
                value = vectors[e][s]
                limit_flag = 0
                if value < self.min_val or value > self.max_val:
                    limit_flag = 1              
                self.encoders[e].compute(value)
            self.pp.compute(learn)
            self.sl.compute(learn)

            if limit_flag == 1:
                anoms.append(1.0)
            else:
                anoms.append(self.sl.get_score())

        self.initialized = True

        return anoms