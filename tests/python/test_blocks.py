from brainblocks.blocks import BlankBlock, ScalarEncoder, SymbolsEncoder, PersistenceEncoder, PatternClassifier, \
    PatternPooler, SequenceLearner
from sklearn import preprocessing
import numpy as np

# printing boolean arrays neatly
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=100,
                    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})


def test_read_write_page():
    blank = BlankBlock(num_s=32)

    # setting and getting bits
    wbits = np.array([0 for i in range(32)])
    wbits[0:4] = 1
    blank.output.bits = wbits
    rbits = np.array(blank.output.bits)
    np.testing.assert_array_equal(wbits, rbits)

    # setting and getting acts
    wacts = np.array([28, 29, 30, 31])
    blank.output.acts = wacts
    racts = np.array(blank.output.acts)
    np.testing.assert_array_equal(wacts, racts)


def test_read_indicator():
    e = ScalarEncoder(min_val=-1.0, max_val=1.0, num_s=128, num_as=16)
    pp = PatternPooler(num_s=128, num_as=2)

    pp.input.add_child(e.output)
    pp.initialize()

    cs = pp.coincidence_sets(0)

    addrs = cs.get_addrs()
    addr0 = cs.get_addr(0)
    np.testing.assert_equal(addrs[0], addr0)

    perms = cs.get_perms()
    perm0 = cs.get_perm(0)
    np.testing.assert_equal(perms[0], perm0)


def test_scalar_encoder():
    e = ScalarEncoder(min_val=-1.0, max_val=1.0, num_s=1024, num_as=128)

    e.compute(-1.5)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[0:128] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(-1.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[0:128] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(-0.5)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[224:352] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[448:576] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.5)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[672:800] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(1.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[896:1024] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(1.5)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[896:1024] = 1
    np.testing.assert_array_equal(actual_out, expect_out)


def test_symbols_encoder():
    # Symbols as strings, converted to integers
    le = preprocessing.LabelEncoder()
    expect_symbols = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
    le.fit(expect_symbols)
    int_labels = le.transform(expect_symbols)

    e = SymbolsEncoder(max_symbols=8, num_s=1024)

    e.compute(int_labels[0])
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[0:128] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(int_labels[1])
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[128:256] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(int_labels[2])
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[256:384] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(int_labels[3])
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[384:512] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(int_labels[4])
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[512:640] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(int_labels[5])
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[640:768] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(int_labels[6])
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[768:896] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(int_labels[7])
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[896:1024] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    try:
        e.compute(int_labels[8])  # exceeds max_symbols limit so brainblocks will throw exception
        assert False, ("Exceeds maximum symbols. An exception should be thrown")
    except:
        pass

    try:
        e.compute(8)  # exceeds max_symbols limit so brainblocks will throw exception
        assert False, ("Exceeds maximum symbols. An exception should be thrown")
    except:
        pass

    actual_symbols = np.array(e.get_symbols())
    np.testing.assert_array_equal(actual_symbols, int_labels)


def test_persistence_encoder():
    e = PersistenceEncoder(min_val=-1.0, max_val=1.0, num_s=1024, num_as=128, max_steps=4)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[0:128] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[0:128] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[224:352] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[448:576] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[672:800] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[896:1024] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[896:1024] = 1
    np.testing.assert_array_equal(actual_out, expect_out)

    e.compute(0.0)
    actual_out = np.array(e.output.get_bits(0))
    expect_out = np.array([0 for i in range(1024)])
    expect_out[896:1024] = 1
    np.testing.assert_array_equal(actual_out, expect_out)


def test_pattern_classifier():
    e = SymbolsEncoder(
        max_symbols=8,
        num_s=1024)

    pc = PatternClassifier(
        labels=(0, 1),
        num_s=512,
        num_as=8,
        perm_thr=20,
        perm_inc=2,
        perm_dec=1,
        pct_pool=0.8,
        pct_conn=0.5,
        pct_learn=0.25)

    pc.input.add_child(e.output)

    for _ in range(10):
        e.compute(0)
        pc.compute(0, True)
        e.compute(1)
        pc.compute(1, True)

    e.compute(0)
    pc.compute(0, False)
    actual_probs = np.array(pc.get_probabilities())
    expect_probs = np.array([1.0, 0.0])
    np.testing.assert_array_equal(actual_probs, expect_probs)

    e.compute(1)
    pc.compute(1, False)
    actual_probs = np.array(pc.get_probabilities())
    expect_probs = np.array([0.0, 1.0])
    np.testing.assert_array_equal(actual_probs, expect_probs)


def test_pattern_pooler():
    e = SymbolsEncoder(
        max_symbols=8,
        num_s=1024)

    pp = PatternPooler(
        num_s=512,
        num_as=8,
        perm_thr=20,
        perm_inc=2,
        perm_dec=1,
        pct_pool=0.8,
        pct_conn=0.5,
        pct_learn=0.25)

    pp.input.add_child(e.output)

    e.compute(0)
    pp.compute(False)
    before_a = pp.output.get_bits()

    e.compute(1)
    pp.compute(False)
    before_b = pp.output.get_bits()

    for _ in range(10):
        e.compute(0)
        pp.compute(True)
        e.compute(1)
        pp.compute(True)

    e.compute(0)
    pp.compute(False)
    after_a = pp.output.get_bits()

    e.compute(1)
    pp.compute(False)
    after_b = pp.output.get_bits()

    np.testing.assert_array_equal(before_a, after_a)
    np.testing.assert_array_equal(before_b, after_b)


def test_sequence_learner():
    e = ScalarEncoder(
        min_val=0.0,
        max_val=1.0,
        num_s=64,
        num_as=8)

    sl = SequenceLearner(
        num_spc=10,
        num_dps=10,
        num_rpd=12,
        d_thresh=6,
        perm_thr=1,
        perm_inc=1,
        perm_dec=0)

    sl.input.add_child(e.output)

    data = [
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0]

    expect_scores = np.array([
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0])

    actual_scores = np.array([0.0 for i in range(len(data))])

    for i in range(len(data)):
        e.compute(data[i])
        sl.compute(True)
        actual_scores[i] = sl.get_score()

    np.testing.assert_array_equal(actual_scores, expect_scores)


if __name__ == "__main__":
    test_read_write_page()
    test_read_indicator()
    test_scalar_encoder()
    test_symbols_encoder()
    test_persistence_encoder()
    test_pattern_classifier()
    test_pattern_pooler()
    test_sequence_learner()