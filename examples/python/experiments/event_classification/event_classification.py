# ==============================================================================
# event_classification.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, SequenceLearner, PatternClassifier
from sklearn import preprocessing
import numpy as np

# ==============================================================================
# Abnormal Event Detector
# ==============================================================================
class AbnormalEventDetector:

    def __init__(self, window=5, cooldown=5):
        self.window = window
        self.cooldown = cooldown
        self.scores = [0.0 for _ in range(self.window)]
        self.idx = 0
        self.avg = 0.0
        self.c = self.cooldown

    def compute(self, score=0.0):
        # update simple moving average
        self.scores[self.idx] = score
        self.avg = sum(self.scores) / self.window
        self.idx += 1
        if self.idx >= self.window:
            self.idx = 0

        # update cooldown counter
        if self.c > 0:
            self.c -= 1

        if self.avg == 1.0 and self.c == 0:
            self.c = self.cooldown
            return 1
        else:
            return 0

# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':

    values = [
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.8, 0.6, 0.4, 0.2,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.8, 0.6, 0.4, 0.2,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        ]

    #labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    new_label = 0

    scores = [0.0 for i in range(len(values))]

    st = ScalarTransformer(
        min_val=0.0, # minimum input value
        max_val=1.0, # maximum input value
        num_s=64,    # number of statelets
        num_as=8)    # number of active statelets

    sl = SequenceLearner(
        num_spc=10, # number of statelets per column
        num_dps=10, # number of coincidence detectors per statelet
        num_rpd=12, # number of receptors per coincidence detector
        d_thresh=6, # coincidence detector threshold
        perm_thr=1, # receptor permanence threshold
        perm_inc=1, # receptor permanence increment
        perm_dec=0) # receptor permanence decrement

    pc = PatternClassifier(
        num_l=20,       # number of labels
        num_s=640,      # number of statelets
        num_as=8,       # number of active statelets
        perm_thr=20,    # receptor permanence threshold
        perm_inc=2,     # receptor permanence increment
        perm_dec=1,     # receptor permanence decrement
        pct_pool=0.8,   # pooling percentage
        pct_conn=0.8,   # initially connected percentage
        pct_learn=0.25) # learn percentage 0.25

    sl.input.add_child(st.output)
    pc.input.add_child(sl.output)

    aed = AbnormalEventDetector(5, 5)

    print('val  scr  lbl  prob  ae  output_active_statelets')

    for i in range(len(values)):

        st.set_value(values[i])
        st.feedforward()
        sl.feedforward(learn=True)
        pc.feedforward(learn=False)

        score = sl.get_anomaly_score()
        probs = pc.get_probabilities()

        abnormal_event = aed.compute(score)

        if abnormal_event:
            for _ in range(50):
                pc.set_label(new_label)
                pc.feedforward(learn=True)
            new_label += 1

        winner = np.argmax(probs)
        winner_str = '-'
        #winner_str = str(winner)
        if probs[winner] >= 0.75:
            winner_str = str(winner)

        sl_acts = '[' + ', '.join(map(str, sl.output.acts)) + ']'
        print('%0.1f  %0.1f  %3s  %0.2f  %2d  %s' % (
            values[i], score, winner_str, probs[winner], abnormal_event,
            sl_acts))
