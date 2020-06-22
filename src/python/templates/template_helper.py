from brainblocks.blocks import *

def get_blocks(configs):
    encoders = []
    pattern_classifier = None
    pattern_pooler = None
    sequence_learner = None

    for config in configs:
        if "block" not in config:
            print("Error: The key \"block\" does not exist in block config")
            exit(1)

        if config["block"] == "scalar_encoder":
            min_val = -1.0
            max_val = 1.0
            num_s = 128
            num_as = 16
            if "min_val" in config: min_val = config["min_val"]
            if "max_val" in config: max_val = config["max_val"]
            if "num_s" in config: num_s = config["num_s"]
            if "num_as" in config: num_as = config["num_as"]
            encoders.append(ScalarEncoder(min_val, max_val, num_s, num_as))

        elif config["block"] == "symbols_encoder":
            max_symbols = 8
            num_s = 128
            if "max_symbols" in config: max_symbols = config["max_symbols"]
            if "num_s" in config: num_s = config["num_s"]
            encoders.append(SymbolsEncoder(max_symbols, num_s))

        elif config["block"] == "persistence_encoder":
            min_val = -1.0
            max_val = 1.0
            num_s = 128
            num_as = 16
            max_steps = 16
            if "min_val" in config: min_val = config["min_val"]
            if "max_val" in config: max_val = config["max_val"]
            if "num_s" in config: num_s = config["num_s"]
            if "num_as" in config: num_as = config["num_as"]
            if "max_steps" in config: max_steps = config["max_steps"]
            encoders.append(PersistenceEncoder(min_val, max_val, num_s, num_as, max_steps))

        elif config["block"] == "pattern_classifier":
            labels = (0, 1)
            num_s = 512
            num_as = 8
            pct_pool = 0.8
            pct_conn = 0.8
            pct_learn = 0.25
            if "labels" in config: labels = config["labels"]
            if "num_s" in config: num_s = config["num_s"]
            if "num_as" in config: num_as = config["num_as"]
            if "pct_pool" in config: pct_pool = config["pct_pool"]
            if "pct_conn" in config: pct_conn = config["pct_conn"]
            if "pct_learn" in config: pct_learn = config["pct_learn"]
            pattern_classifier = PatternClassifier(labels, len(labels), num_s, num_as, pct_pool, pct_conn, pct_learn)

        elif config["block"] == "pattern_pooler":
            num_s = 512
            num_as = 8
            pct_pool = 0.8
            pct_conn = 0.8
            pct_learn = 0.25
            if "num_s" in config: num_s = config["num_s"]
            if "num_as" in config: num_as = config["num_as"]
            if "pct_pool" in config: pct_pool = config["pct_pool"]
            if "pct_conn" in config: pct_conn = config["pct_conn"]
            if "pct_learn" in config: pct_learn = config["pct_learn"]
            pattern_pooler = PatternPooler(num_s, num_as, pct_pool, pct_conn, pct_learn)

        elif config["block"] == "sequence_learner":
            num_spc = 10
            num_dps = 10
            num_rpd = 12
            d_thresh = 6
            if "num_spc" in config: num_spc = config["num_spc"]
            if "num_dps" in config: num_dps = config["num_dps"]
            if "num_rpd" in config: num_rpd = config["num_rpd"]
            if "d_thresh" in config: d_thresh = config["d_thresh"]
            sequence_learner = SequenceLearner(num_spc, num_dps, num_rpd, d_thresh)

        else:
            print("Warning: Block type \"%s\" does not exist" % (config["block"]))

    res_dict = {}
    res_dict["encoders"] = encoders
    res_dict["pattern_classifier"] = pattern_classifier
    res_dict["pattern_pooler"] = pattern_pooler
    res_dict["sequence_learner"] = sequence_learner

    return res_dict