import brainblocks.bb_backend as bb
import numpy as np

# seed for deterministic random generator
#bb.seed(0)

# ==============================================================================
# BitArray
# ==============================================================================
class BitArray():
    def __init__(self, bitarray_obj):
        self.obj = bitarray_obj

    def print_info(self):
        self.obj.print_info()

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
# Page
# ==============================================================================
class Page():
    def __init__(self, page_obj):
        self.obj = page_obj

    def __getitem__(self, t):
        return BitArray(self.obj[t])

    def print_info(self):
        self.obj.print_info()

    def add_child(self, child_page):
        self.obj.add_child(child_page.obj)

    #def get_child(self, child_index):
    #    return Page(self.obj.get_child(child_index))

    @property
    def num_bitarrays(self):
        return self.obj.num_bitarrays

    @property
    def num_children(self):
        return self.obj.num_children

# ==============================================================================
# CoincidenceSet
# ==============================================================================
class CoincidenceSet():
    def __init__(self, coincidence_set_obj):
        self.obj = coincidence_set_obj

    def get_addr(self, r):
        return self.obj.get_addr(r)

    def get_perm(self, r):
        return self.obj.get_perm(r)

    def get_addrs(self):
        return self.obj.get_addrs()

    def get_perms(self):
        return self.obj.get_perms()

    #def get_bits(self):
    #    return self.obj.get_bits()

    #def get_acts(self):
    #    return self.obj.get_acts()

    #bits = property(get_bits, None, None, "Read Bits")
    #acts = property(get_acts, None, None, "Read Acts")

# ==============================================================================
# BlankBlock
# ==============================================================================
class BlankBlock():
    def __init__(
            self,
            num_s=1024): # number of statelets

        self.obj = bb.BlankBlock(num_s)

    def clear_states(self):
        self.obj.clear_states()

    @property
    def output(self):
        return Page(self.obj.output)

# ==============================================================================
# ScalarEncoder
# ==============================================================================
class ScalarEncoder():
    def __init__(
            self,
            min_val=-1.0, # minimum input value
            max_val=1.0,  # maximum input value
            num_s=1024,   # number of statelets
            num_as=128):  # number of active statelets

        self.obj = bb.ScalarEncoder(min_val, max_val, num_s, num_as)

    def initialize(self):
        self.obj.initialize()

    def clear_states(self):
        self.obj.clear_states()

    def compute(self, value):
        if value != None:
            self.obj.compute(value)

    @property
    def output(self):
        return Page(self.obj.output)

# ==============================================================================
# SymbolsEncoder
# ==============================================================================
class SymbolsEncoder():
    def __init__(
            self,
            max_symbols=8, # maximum number of symbols
            num_s=1024):   # number of statelets

        self.obj = bb.SymbolsEncoder(max_symbols, num_s)

    def initialize(self):
        self.obj.initialize()

    def clear_states(self):
        self.obj.clear_states()

    def compute(self, value):
        if value != None:
            self.obj.compute(value)

    def get_symbols(self):
        return self.obj.get_symbols()

    @property
    def output(self):
        return Page(self.obj.output)

# ==============================================================================
# PersistenceEncoder
# ==============================================================================
class PersistenceEncoder():
    def __init__(
            self,
            min_val=-1.0,   # minimum input value
            max_val=1.0,    # maximum input value
            num_s=1024,     # number of statelets
            num_as=128,     # number of active statelets
            max_steps=100): # maximum number of persistence steps

        self.obj = bb.PersistenceEncoder(min_val, max_val, num_s, num_as, max_steps)

    def initialize(self):
        self.obj.initialize()

    def clear_states(self):
        self.obj.clear()

    def compute(self, value):
        if value != None:
            self.obj.compute(value)

    @property
    def output(self):
        return Page(self.obj.output)

# ==============================================================================
# PatternClassifier
# ==============================================================================
class PatternClassifier():
    def __init__(
            self,
            labels=(0,1),    # user-defined labels
            num_s=512,       # number of statelets
            num_as=8,        # number of active statelets
            perm_thr=20,     # receptor permanence threshold
            perm_inc=2,      # receptor permanence increment
            perm_dec=1,      # receptor permanence decrement
            pct_pool=0.8,    # pooling percentage
            pct_conn=0.5,    # initially connected percentage
            pct_learn=0.25,  # learn percentage
            random_state=0): # random state integer

        #if isinstance(random_state,int): # TODO: fix seeding
            #bb.seed(random_state)

        self.obj = bb.PatternClassifier(
            labels, num_s, num_as,
            perm_thr, perm_inc, perm_dec,
            pct_pool, pct_conn, pct_learn)

    def initialize(self):
        self.obj.initialize()

    def save(self, file_str='./pc.bin'):
        self.obj.save(file_str.encode('utf-8'))

    def load(self, file_str='./pc.bin'):
        self.obj.load(file_str.encode('utf-8'))

    def clear_states(self):
        self.obj.clear_states()

    def compute(self, label=None, learn=False):
        if learn:
            if label is None:
                raise ("Label required for PatternClassifier when learn=True")

            if not np.issubdtype(type(label), np.integer):
                raise ("Label must be of type integer for PatternClassifier")

            self.obj.compute(label, 1)

        else:
            self.obj.compute(0, 0)

    def get_labels(self):
        return self.obj.get_labels()

    def get_probabilities(self):
        return self.obj.get_probabilities()

    def output_coincidence_set(self, d):
        return CoincidenceSet(self.obj.output_coincidence_set(d))

    @property
    def input(self):
        return Page(self.obj.input)

    @property
    def output(self):
        return Page(self.obj.output)

# ==============================================================================
# PatternPooler
# ==============================================================================
class PatternPooler():
    def __init__(
            self,
            num_s=512,       # number of statelets
            num_as=8,        # number of active statelets
            perm_thr=20,     # receptor permanence threshold
            perm_inc=2,      # receptor permanence increment
            perm_dec=1,      # receptor permanence decrement
            pct_pool=0.8,    # pooling percentage
            pct_conn=0.5,    # initially connected percentage
            pct_learn=0.25,  # learn percentage
            random_state=0): # random state integer

        #if isinstance(random_state,int): # TODO: fix seeding
        #    bb.seed(random_state)

        self.obj = bb.PatternPooler(
            num_s, num_as, 
            perm_thr, perm_inc, perm_dec,
            pct_pool, pct_conn, pct_learn)

    def initialize(self):
        self.obj.initialize()

    def save(self, file_str='./pl.bin'):
        self.obj.save(file_str.encode('utf-8'))

    def load(self, file_str='./pl.bin'):
        self.obj.load(file_str.encode('utf-8'))

    def clear_states(self):
        self.obj.clear()

    def compute(self, learn=True):
        self.obj.compute(learn)

    def output_coincidence_set(self, d):
        return CoincidenceSet(self.obj.output_coincidence_set(d))

    @property
    def input(self):
        return Page(self.obj.input)

    @property
    def output(self):
        return Page(self.obj.output)

# ==============================================================================
# SequenceLearner
# ==============================================================================
class SequenceLearner():
    def __init__(
            self,
            num_spc=10,      # number of statelets per column
            num_dps=10,      # number of coincidence detectors per statelet
            num_rpd=12,      # number of receptors per coincidence detector
            d_thresh=6,      # coincidence detector threshold
            perm_thr=20,     # receptor permanence threshold
            perm_inc=2,      # receptor permanence increment
            perm_dec=1,      # receptor permanence decrement
            random_state=0): # random state integer

        #if isinstance(random_state,int): # TODO: fix seeding
        #    bb.seed(random_state)

        self.obj = bb.SequenceLearner(
            num_spc, num_dps, num_rpd, d_thresh,
            perm_thr, perm_inc, perm_dec)

    def initialize(self):
        self.obj.initialize()

    def save(self, file_str='./obj.bin'):
        self.obj.save(file_str.encode('utf-8'))

    def load(self, file_str='./obj.bin'):
        self.obj.load(file_str.encode('utf-8'))

    def clear_states(self):
        self.obj.clear_states()

    def compute(self, learn=True):
        self.obj.compute(learn)

    def get_score(self):
        return self.obj.get_score()

    #def get_historical_count(self):
    #    return self.obj.get_historical_count()

    #def get_coincidence_set_count(self):
    #    return self.obj.get_coincidence_set_count()

    #def get_historical_statelets(self):
    #    return self.obj.get_historical_statelets()

    #def get_num_coincidence_sets_per_statelet(self):
    #    return self.obj.get_num_coincidence_sets_per_statelet()

    #def print_column(self, c):
    #    self.obj.print_column(c)

    def hidden_coincidence_set(self, d):
        return CoincidenceSet(self.obj.hidden_coincidence_set(d))

    def output_coincidence_set(self, d):
        return CoincidenceSet(self.obj.output_coincidence_set(d))

    @property
    def input(self):
        return Page(self.obj.input)

    @property
    def hidden(self):
        return Page(self.obj.hidden)

    @property
    def output(self):
        return Page(self.obj.output)