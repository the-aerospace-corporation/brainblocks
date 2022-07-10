import brainblocks.bb_backend as bb
import numpy as np

# ==============================================================================
# BitArray
# ==============================================================================
class BitArray():

    def __init__(self, bitarray_obj):
        self.obj = bitarray_obj

    def set_bits(self, new_bits=[]):
        self.obj.set_bits(new_bits)

    def set_acts(self, new_acts=[]):
        self.obj.set_acts(new_acts)

    def get_bits(self):
        return self.obj.get_bits()

    def get_acts(self):
        return self.obj.get_acts()

    @property
    def num_bits(self):
        return self.obj.num_bits

    @property
    def num_words(self):
        return self.obj.num_words

    bits = property(get_bits, set_bits, None, "Read/Write Bits")
    acts = property(get_acts, set_acts, None, "Read/Write Acts")

# ==============================================================================
# BlockInput
# ==============================================================================
class BlockInput():

    def __init__(self, block_input_obj):
        self.obj = block_input_obj

    def add_child(self, src, t=0):
        self.obj.add_child(src.obj, t)

    @property
    def num_children(self):
        return self.obj.num_children

    @property
    def state(self):
        return BitArray(self.obj.state)

    @property
    def bits(self):
        return BitArray(self.obj.state).bits

    @bits.setter
    def bits(self, new_bits=[]):
        BitArray(self.obj.state).bits = new_bits

    @property
    def acts(self):
        return BitArray(self.obj.state).acts

    @acts.setter
    def acts(self, new_acts=[]):
        BitArray(self.obj.state).acts = new_acts

    @property
    def num_bits(self):
        return BitArray(self.obj.state).num_bits

    @property
    def num_words(self):
        return BitArray(self.obj.state).num_words

# ==============================================================================
# BlockMemory
# ==============================================================================
class BlockMemory():

    def __init__(self, block_memory_obj):
        self.obj = block_memory_obj

    def addrs(self, d):
        return self.obj.addrs(d)

    def perms(self, d):
        return self.obj.perms(d)

    def conns(self, d):
        return self.obj.conns(d)

# ==============================================================================
# BlockOutput
# ==============================================================================
class BlockOutput():

    def __init__(self, block_output_obj):
        self.obj = block_output_obj

    def __getitem__(self, t):
        return BitArray(self.obj.get_bitarray(t))

    @property
    def num_t(self):
        return self.obj.num_t

    @property
    def state(self):
        return BitArray(self.obj.state)

    @property
    def bits(self):
        return BitArray(self.obj.state).bits

    @bits.setter
    def bits(self, new_bits=[]):
        BitArray(self.obj.state).bits = new_bits

    @property
    def acts(self):
        return BitArray(self.obj.state).acts

    @acts.setter
    def acts(self, new_acts=[]):
        BitArray(self.obj.state).acts = new_acts

    @property
    def num_bits(self):
        return BitArray(self.obj.state).num_bits

    @property
    def num_words(self):
        return BitArray(self.obj.state).num_words

# ==============================================================================
# BlankBlock
# ==============================================================================
class BlankBlock():

    def __init__(
            self,
            num_s=512, # number of statelets
            num_t=2,  # number of BlockOutput time steps (optional)
            seed=0):     # seed for random number generator

        self.obj = bb.BlankBlock(num_s, num_t)

    def feedforward(self, learn_flag=False):
        self.obj.feedforward(learn_flag)

    @property
    def output(self):
        return BlockOutput(self.obj.output)

# ==============================================================================
# ContextLearner
# ==============================================================================
class ContextLearner():

    def __init__(
            self,
            num_c=512,   # number of columns
            num_spc=10,  # number of statelets per column
            num_dps=10,  # number of coincidence detectors per statelet
            num_rpd=12,  # number of receptors per coincidence detector
            d_thresh=6,  # coincidence detector threshold
            perm_thr=20, # receptor permanence threshold
            perm_inc=2,  # receptor permanence increment
            perm_dec=1,  # receptor permanence decrement
            num_t=2,     # number of BlockOutput time steps (optional)
            seed=0):     # seed for random number generator

        self.obj = bb.ContextLearner(
            num_c, num_spc, num_dps, num_rpd, d_thresh, perm_thr, perm_inc,
            perm_dec, num_t, seed)

    def init(self):
        self.obj.init()

    def save(self, file='./file.bin'):
        return self.obj.save(file.encode('utf-8'))

    def load(self, file='./file.bin'):
        return self.obj.load(file.encode('utf-8'))

    def clear(self):
        self.obj.clear()

    def feedforward(self, learn=False):
        self.obj.feedforward(learn)

    def get_anomaly_score(self):
        return self.obj.get_anomaly_score()

    @property
    def input(self):
        return BlockInput(self.obj.input)

    @property
    def context(self):
        return BlockInput(self.obj.context)

    @property
    def output(self):
        return BlockOutput(self.obj.output)

    @property
    def memory(self):
        return BlockMemory(self.obj.memory)

# ==============================================================================
# DiscreteTransformer
# ==============================================================================
class DiscreteTransformer():
    def __init__(
            self,
            num_v=8,   # number of discrete values
            num_s=512, # number of statelets
            num_t=2,  # number of BlockOutput time states (optional)
            seed=0):     # seed for random number generator

        self.obj = bb.DiscreteTransformer(num_v, num_s, num_t)

    def clear(self):
        self.obj.clear()

    def feedforward(self):
        self.obj.feedforward()

    def set_value(self, value):
        self.obj.set_value(value)

    def get_value(self):
        return self.obj.get_value()

    @property
    def output(self):
        return BlockOutput(self.obj.output)

# ==============================================================================
# PatternClassifier
# ==============================================================================
class PatternClassifier():

    def __init__(
            self,
            num_l,         # number of labels
            num_s=512,     # number of statelets
            num_as=8,      # number of active statelets
            perm_thr=20,   # receptor permanence threshold
            perm_inc=2,    # receptor permanence increment
            perm_dec=1,    # receptor permanence decrement
            pct_pool=0.8,  # pooling percentage
            pct_conn=0.5,  # initially connected percentage
            pct_learn=0.3, # learn percentage
            num_t=2,       # number of BlockOutput time steps (optional)
            seed=0):       # seed for random number generator

        self.obj = bb.PatternClassifier(
            num_l, num_s, num_as, perm_thr, perm_inc, perm_dec, pct_pool,
            pct_conn, pct_learn, num_t, seed)

    def init(self):
        self.obj.init()

    def save(self, file='./file.bin'):
        return self.obj.save(file.encode('utf-8'))

    def load(self, file='./file.bin'):
        return self.obj.load(file.encode('utf-8'))

    def clear(self):
        self.obj.clear()

    def feedforward(self, learn=False):
        self.obj.feedforward(learn)

    def set_label(self, label):
        self.obj.set_label(label)

    def get_labels(self):
        return self.obj.get_labels()

    def get_probabilities(self):
        return self.obj.get_probabilities()

    @property
    def input(self):
        return BlockInput(self.obj.input)

    @property
    def output(self):
        return BlockOutput(self.obj.output)

    @property
    def memory(self):
        return BlockMemory(self.obj.memory)

# ==============================================================================
# PatternClassifierDynamic
# ==============================================================================
class PatternClassifierDynamic():

    def __init__(
            self,
            num_s=512,     # number of statelets
            num_as=8,      # number of active statelets
            num_spl=32,    # number of statelets per label
            perm_thr=20,   # receptor permanence threshold
            perm_inc=2,    # receptor permanence increment
            perm_dec=1,    # receptor permanence decrement
            pct_pool=0.8,  # pooling percentage
            pct_conn=0.5,  # initially connected percentage
            pct_learn=0.3, # learn percentage
            num_t=2,       # number of BlockOutput time steps (optional)
            seed=0):       # seed for random number generator

        self.obj = bb.PatternClassifierDynamic(
            num_s, num_as, num_spl, perm_thr, perm_inc, perm_dec, pct_pool,
            pct_conn, pct_learn, num_t, seed)

    def init(self):
        self.obj.init()

    def save(self, file='./file.bin'):
        return self.obj.save(file.encode('utf-8'))

    def load(self, file='./file.bin'):
        return self.obj.load(file.encode('utf-8'))

    def clear(self):
        self.obj.clear()

    def feedforward(self, learn=False):
        self.obj.feedforward(learn)

    def set_label(self, label):
        self.obj.set_label(label)

    def get_anomaly_score(self):
        return self.obj.get_anomaly_score()

    def get_labels(self):
        return self.obj.get_labels()

    def get_probabilities(self):
        return self.obj.get_probabilities()

    @property
    def input(self):
        return BlockInput(self.obj.input)

    @property
    def output(self):
        return BlockOutput(self.obj.output)

    @property
    def memory(self):
        return BlockMemory(self.obj.memory)

# ==============================================================================
# PatternPooler
# ==============================================================================
class PatternPooler():

    def __init__(
            self,
            num_s=512,     # number of statelets
            num_as=8,      # number of active statelets
            perm_thr=20,   # receptor permanence threshold
            perm_inc=2,    # receptor permanence increment
            perm_dec=1,    # receptor permanence decrement
            pct_pool=0.8,  # percent pooled
            pct_conn=0.5,  # percent initially connected
            pct_learn=0.3, # percent learn
            num_t=2,       # number of BlockOutput time steps (optional)
            always_update=False,  # whether to update when the input doesn't change
            seed=0):       # seed for random number generator

        self.obj = bb.PatternPooler(
            num_s, num_as, perm_thr, perm_inc, perm_dec, pct_pool, pct_conn,
            pct_learn, num_t, always_update, seed)

    def init(self):
        self.obj.init()

    def save(self, file='./file.bin'):
        return self.obj.save(file.encode('utf-8'))

    def load(self, file='./file.bin'):
        return self.obj.load(file.encode('utf-8'))

    def clear(self):
        self.obj.clear()

    def feedforward(self, learn=False):
        self.obj.feedforward(learn)

    @property
    def input(self):
        return BlockInput(self.obj.input)

    @property
    def output(self):
        return BlockOutput(self.obj.output)

    @property
    def memory(self):
        return BlockMemory(self.obj.memory)

# ==============================================================================
# PersistenceTransformer
# ==============================================================================
class PersistenceTransformer():

    def __init__(
            self,
            min_val=-1.0, # minimum input value
            max_val=1.0,  # maximum input value
            num_s=512,    # number of statelets
            num_as=64,    # number of active statelets
            max_step=10,  # maximum number of persistence steps
            num_t=2,     # number of BlockOutput time steps (optional)
            seed=0):     # seed for random number generator

        self.obj = bb.PersistenceTransformer(
            min_val, max_val, num_s, num_as, max_step, num_t)

    def clear(self):
        self.obj.clear()

    def feedforward(self):
        self.obj.feedforward()

    def set_value(self, value):
        self.obj.set_value(value)

    def get_value(self):
        return self.obj.get_value()

    @property
    def output(self):
        return BlockOutput(self.obj.output)

# ==============================================================================
# ScalarTransformer
# ==============================================================================
class ScalarTransformer():

    def __init__(
            self,
            min_val=-1.0, # minimum input value
            max_val=1.0,  # maximum input value
            num_s=512,    # number of statelets
            num_as=64,    # number of active statelets
            num_t=2,     # number of BlockOutput time steps (optional)
            seed=0):     # seed for random number generator

        self.obj = bb.ScalarTransformer(
            min_val, max_val, num_s, num_as, num_t)

    def clear(self):
        self.obj.clear()

    def feedforward(self):
        self.obj.feedforward()

    def set_value(self, value):
        self.obj.set_value(value)

    def get_value(self):
        return self.obj.get_value()

    @property
    def output(self):
        return BlockOutput(self.obj.output)

# ==============================================================================
# SequenceLearner
# ==============================================================================
class SequenceLearner():

    def __init__(
            self,
            num_c=512,   # number of columns
            num_spc=10,  # number of statelets per column
            num_dps=10,  # number of coincidence detectors per statelet
            num_rpd=12,  # number of receptors per coincidence detector
            d_thresh=6,  # coincidence detector threshold
            perm_thr=20, # receptor permanence threshold
            perm_inc=2,  # receptor permanence increment
            perm_dec=1,  # receptor permanence decrement
            num_t=2,     # number of BlockOutput time steps (optional)
            always_update=False,  # whether to update when the input doesn't change
            seed=0):     # seed for random number generator

        self.obj = bb.SequenceLearner(
            num_c, num_spc, num_dps, num_rpd, d_thresh, perm_thr, perm_inc,
            perm_dec, num_t, always_update, seed)

    def init(self):
        self.obj.init()

    def save(self, file='./file.bin'):
        return self.obj.save(file.encode('utf-8'))

    def load(self, file='./file.bin'):
        return self.obj.load(file.encode('utf-8'))

    def clear(self):
        self.obj.clear()

    def feedforward(self, learn=False):
        self.obj.feedforward(learn)

    def get_anomaly_score(self):
        return self.obj.get_anomaly_score()

    def get_historical_count(self):
        return self.obj.get_historical_count()

    @property
    def input(self):
        return BlockInput(self.obj.input)

    @property
    def context(self):
        return BlockInput(self.obj.context)

    @property
    def output(self):
        return BlockOutput(self.obj.output)

    @property
    def memory(self):
        return BlockMemory(self.obj.memory)
