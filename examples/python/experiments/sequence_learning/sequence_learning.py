# ==============================================================================
# sequence_learning.py
#
# TODO: this is out of data and needs updating!!!
# ==============================================================================
import os
import errno
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import brainblocks.bb_backend as bb
from brainblocks.blocks import SymbolsEncoder, SequenceLearner
from sklearn import preprocessing

# seed for deterministic random generator
bb.seed(0)

# printing boolean arrays neatly
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=100,
                    formatter={'bool': lambda bin_val: 'X' if bin_val else '-'})

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

# ==============================================================================
# Plot Results
# ==============================================================================
def plot_results(directory, title, values, scores, total_statelets, total_hidden_statelets, total_historical, total_coincidence_sets, s_upper_limit, cs_upper_limit):
    t = [i for i in range(len(values))]

    plt.clf()
    fig, axes = plt.subplots(4, 1, num=1, sharex=True)
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    ax3 = axes[3]

    # axis 0: values
    ax0.plot(t, values, 'o-', drawstyle='steps-mid', linewidth=0.5, markersize=3)
    ax0.set_ylabel('symbols')

    # axis 1: scores
    ax1.plot(t, scores, drawstyle='steps-mid')
    ax1.set_ylabel('score')

    # axis 2: statelets
    ax2.plot(t, total_statelets, drawstyle='steps-mid', label='output')
    ax2.plot(t, total_hidden_statelets, drawstyle='steps-mid', label='hidden')
    #ax2.fill_between(t, total_historical, 0, alpha=0.2, label='historical')
    ax2.set_ylabel('statelets')
    #ax2.set_ylim(0, s_upper_limit)
    #ax2.legend(loc='upper left')

    # create the legend on the figure, not the axes
    handles, labels = ax2.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right',
    #              bbox_to_anchor=(1.1, 0.9))  # , fontsize=20)

    fig.legend(handles, labels, fontsize=8, title_fontsize=8, bbox_to_anchor=(1.15, 1),
               bbox_transform=ax2.transAxes)

    # axis 3: coincidence sets
    ax3.plot(t, total_coincidence_sets, drawstyle='steps-mid')
    ax3.set_ylabel('coincidence sets')
    ax3.set_ylim(0, cs_upper_limit)

    # save plot
    #fig.suptitle()
    plt.savefig('./%s/%s.png' % (directory, title))
    plt.close(fig)

# ==============================================================================
# Plot Statelets
# ==============================================================================
def plot_statelets(directory, title, statelets):
    len_statelets = len(statelets)
    sroot = math.ceil(math.sqrt(len_statelets))

    data = [[0] * sroot for _ in range(sroot)]

    for i in range(len_statelets):
        y = int(i / sroot)
        x = int(i % sroot)
        data[x][y] = statelets[i]

    # create discrete colormap
    cmap = mpl.colors.ListedColormap(['white', 'black'])
    bounds = [0.0, 0.5, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

    fig.suptitle(title)
    plt.savefig('./%s/%s.png' % (directory, title))
    plt.close(fig)

# ==============================================================================
# Plot Statelet Usage
# ==============================================================================
def plot_statelet_usage(directory, title, statelets, vmax=None):
    len_statelets = len(statelets)
    sroot = math.ceil(math.sqrt(len_statelets))

    data = [[0] * sroot for _ in range(sroot)]

    for i in range(len_statelets):
        y = int(i / sroot)
        x = int(i % sroot)
        data[x][y] = statelets[i]

    fig, ax = plt.subplots()
    ax.imshow(data, cmap='viridis', vmax=vmax)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

    fig.suptitle(title + ' statelet usage')
    plt.savefig('./%s/statelet_usage_%s.png' % (directory, title))
    plt.close(fig)

# ==============================================================================
# One Event
# ==============================================================================
def one_event(statelet_snapshots_on=False):
    experiment_name = 'one_event'
    directory = './' + experiment_name
    mkdir_p(directory)
    print()
    print('experiment=\'%s\'' % (experiment_name))

    e = SymbolsEncoder(
        max_symbols=26, # maximum number of symbols
        num_s=208)      # number of statelets

    sl = SequenceLearner(
        num_spc=10, # number of statelets per column
        num_dps=50, # number of coincidence detectors per statelet
        num_rpd=12, # number of receptors per coincidence detector
        d_thresh=6, # coincidence detector threshold
        perm_thr=1, # receptor permanence threshold
        perm_inc=1, # receptor permanence increment
        perm_dec=0) # receptor permanence decrement

    sl.input.add_child(e.output)

    values = [
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'b', 'c', 'd', 'e', 'f',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'b', 'c', 'd', 'e', 'f',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'b', 'c', 'd', 'e', 'f',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a']

    le = preprocessing.LabelEncoder()
    le.fit(values)
    int_values = le.transform(values)

    scores = [0.0 for _ in range(len(values))]
    count_s_output_acts = [0 for _ in range(len(values))]
    count_s_hidden_acts = [0 for _ in range(len(values))]
    count_s_hist = [0 for _ in range(len(values))]
    count_cs = [0 for _ in range(len(values))]
    hidden_s_usage = [0 for _ in range(2240)]
    output_s_usage = [0 for _ in range(2240)]

    print('val  scr  s_act  s_his    cs  active output statelets')

    for i in range(len(int_values)):
        e.compute(value=int_values[i])
        sl.compute(learn=True)

        # update information
        hidden_s_bits = sl.hidden.bits
        hidden_s_acts = sl.hidden.acts
        output_s_bits = sl.output.bits
        output_s_acts = sl.output.acts
        scores[i] = sl.get_score()
        count_s_output_acts[i] = len(output_s_acts)
        count_s_hidden_acts[i] = len(hidden_s_acts)
        count_s_hist[i] = sl.get_historical_count()
        count_cs[i] = sl.get_coincidence_set_count()

        # update statelet usage
        for s in range(len(output_s_usage)):
            hidden_s_usage[s] += hidden_s_bits[s]
            output_s_usage[s] += output_s_bits[s]

        # plot statelets
        if statelet_snapshots_on and (i+1) % 5 == 0:
            title = 'step_' + str(i) + '_' + values[i] + '_' + values[i-1]
            plot_statelets(directory, 'hidden_'+title, hidden_s_bits)
            plot_statelets(directory, 'output_'+title, output_s_bits)

        # print information
        output_s_acts_str = '[' + ', '.join(str(act).rjust(4) for act in output_s_acts) + ']'
        print('{0:>3}  {1:0.1f}  {2:5d}  {3:5d}  {4:4d}  {5:>4}'.format(
            values[i], scores[i], count_s_output_acts[i], count_s_hist[i], count_cs[i], output_s_acts_str))

    # plot information
    plot_results(directory, 'results', values, scores, count_s_output_acts, count_s_hidden_acts, count_s_hist, count_cs, 400, 400)
    plot_statelet_usage(directory, 'hidden', hidden_s_usage, 40)
    plot_statelet_usage(directory, 'output', output_s_usage, 40)

# ==============================================================================
# Two Events
# ==============================================================================
def two_events(statelet_snapshots_on=False):
    experiment_name = 'two_events'
    directory = './' + experiment_name
    mkdir_p(directory)
    print()
    print('experiment=\'%s\'' % (experiment_name))

    e = SymbolsEncoder(
        max_symbols=26, # maximum number of symbols
        num_s=208)      # number of statelets

    sl = SequenceLearner(
        num_spc=10, # number of statelets per column
        num_dps=50, # number of coincidence detectors per statelet
        num_rpd=12, # number of receptors per coincidence detector
        d_thresh=6, # coincidence detector threshold
        perm_thr=1, # receptor permanence threshold
        perm_inc=1, # receptor permanence increment
        perm_dec=0) # receptor permanence decrement

    sl.input.add_child(e.output)

    values = [
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'b', 'c', 'd', 'e', 'f',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'f', 'e', 'd', 'c', 'b',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'b', 'c', 'd', 'e', 'f',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'f', 'e', 'd', 'c', 'b',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'b', 'c', 'd', 'e', 'f',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'f', 'e', 'd', 'c', 'b',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a']

    le = preprocessing.LabelEncoder()
    le.fit(values)
    int_values = le.transform(values)

    scores = [0.0 for _ in range(len(values))]
    count_s_output_acts = [0 for _ in range(len(values))]
    count_s_hidden_acts = [0 for _ in range(len(values))]
    count_s_hist = [0 for _ in range(len(values))]
    count_cs = [0 for _ in range(len(values))]
    hidden_s_usage = [0 for _ in range(2240)]
    output_s_usage = [0 for _ in range(2240)]

    print('val  scr  s_act  s_his    cs  active output statelets')

    for i in range(len(int_values)):
        e.compute(value=int_values[i])
        sl.compute(learn=True)

        # update information
        hidden_s_bits = sl.hidden.bits
        hidden_s_acts = sl.hidden.acts
        output_s_bits = sl.output.bits
        output_s_acts = sl.output.acts
        scores[i] = sl.get_score()
        count_s_output_acts[i] = len(output_s_acts)
        count_s_hidden_acts[i] = len(hidden_s_acts)
        count_s_hist[i] = sl.get_historical_count()
        count_cs[i] = sl.get_coincidence_set_count()

        # update statelet usage
        for s in range(len(output_s_usage)):
            hidden_s_usage[s] += hidden_s_bits[s]
            output_s_usage[s] += output_s_bits[s]

        # plot statelets
        if statelet_snapshots_on and (i+1) % 5 == 0:
            title = 'step_' + str(i) + '_' + values[i] + '_' + values[i-1]
            plot_statelets(directory, 'hidden_'+title, hidden_s_bits)
            plot_statelets(directory, 'output_'+title, output_s_bits)

        # print information
        output_s_acts_str = '[' + ', '.join(str(act).rjust(4) for act in output_s_acts) + ']'
        print('{0:>3}  {1:0.1f}  {2:5d}  {3:5d}  {4:4d}  {5:>4}'.format(
            values[i], scores[i], count_s_output_acts[i], count_s_hist[i], count_cs[i], output_s_acts_str))

    # plot information
    plot_results(directory, 'results', values, scores, count_s_output_acts, count_s_hidden_acts, count_s_hist, count_cs, 400, 400)
    plot_statelet_usage(directory, 'hidden', hidden_s_usage, 75)
    plot_statelet_usage(directory, 'output', output_s_usage, 75)

# ==============================================================================
# Three Events
# ==============================================================================
def three_events(statelet_snapshots_on=False):
    experiment_name = 'three_events'
    directory = './' + experiment_name
    mkdir_p(directory)
    print()
    print('experiment=\'%s\'' % (experiment_name))

    NUM_S = 208
    NUM_SPC = 10
    TOTAL_NUM_S = NUM_S * NUM_SPC


    e = SymbolsEncoder(
        max_symbols=26,  # maximum number of symbols
        num_s=NUM_S)  # number of statelets

    sl = SequenceLearner(
        num_spc=NUM_SPC,  # number of statelets per column
        num_dps=50,  # number of coincidence detectors per statelet
        num_rpd=12,  # number of receptors per coincidence detector
        d_thresh=6,  # coincidence detector threshold
        perm_thr=1,  # receptor permanence threshold
        perm_inc=1,  # receptor permanence increment
        perm_dec=0)  # receptor permanence decrement

    sl.input.add_child(e.output)

    values = [
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'b', 'c', 'd', 'e', 'f',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'f', 'e', 'd', 'c', 'b',
        'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a',
        'b', 'c', 'd', 'c', 'b',
        'a', 'a', 'a', 'a', 'a']

    le = preprocessing.LabelEncoder()
    le.fit(values)
    int_values = le.transform(values)

    scores = [0.0 for _ in range(len(values))]
    count_s_output_acts = [0 for _ in range(len(values))]
    count_s_hidden_acts = [0 for _ in range(len(values))]
    count_s_hist = [0 for _ in range(len(values))]
    count_cs = [0 for _ in range(len(values))]
    hidden_s_usage = [0 for _ in range(TOTAL_NUM_S)]
    output_s_usage = [0 for _ in range(TOTAL_NUM_S)]

    print('val  scr  s_act  s_his    cs  active output statelets')

    for i in range(len(int_values)):
        e.compute(value=int_values[i])
        sl.compute(learn=True)

        # update information
        hidden_s_bits = sl.hidden.bits
        hidden_s_acts = sl.hidden.acts
        output_s_bits = sl.output.bits
        output_s_acts = sl.output.acts
        scores[i] = sl.get_score()
        count_s_output_acts[i] = len(output_s_acts)
        count_s_hidden_acts[i] = len(hidden_s_acts)
        count_s_hist[i] = sl.get_historical_count()
        count_cs[i] = sl.get_coincidence_set_count()

        # update statelet usage
        for s in range(len(output_s_usage)):
            hidden_s_usage[s] += hidden_s_bits[s]
            output_s_usage[s] += output_s_bits[s]

        # plot statelets
        if statelet_snapshots_on and (i+1) % 5 == 0 or i == 43:
            title = 'step_' + str(i) + '_' + values[i] + '_' + values[i-1]
            plot_statelets(directory, 'hidden_'+title, hidden_s_bits)
            plot_statelets(directory, 'output_'+title, output_s_bits)

        # print information
        output_s_acts_str = '[' + ', '.join(str(act).rjust(4) for act in output_s_acts) + ']'
        print('{0:>3}  {1:0.1f}  {2:5d}  {3:5d}  {4:4d}  {5:>4}'.format(
            values[i], scores[i], count_s_output_acts[i], count_s_hist[i], count_cs[i], output_s_acts_str))

    # plot information
    plot_results(directory, 'results', values, scores, count_s_output_acts, count_s_hidden_acts, count_s_hist, count_cs, 400, 400)
    plot_statelet_usage(directory, 'hidden', hidden_s_usage, 75)
    plot_statelet_usage(directory, 'output', output_s_usage, 75)

# ==============================================================================
# Multiple Prior Contexts
# ==============================================================================
def multiple_prior_contexts(statelet_snapshots_on=False):
    experiment_name = 'multiple_prior_contexts'
    directory = './' + experiment_name
    mkdir_p(directory)
    print()
    print('experiment=\'%s\'' % (experiment_name))

    e = SymbolsEncoder(
        max_symbols=26, # maximum number of symbols
        num_s=208)      # number of statelets

    sl = SequenceLearner(
        num_spc=10, # number of statelets per column
        num_dps=50, # number of coincidence detectors per statelet
        num_rpd=12, # number of receptors per coincidence detector
        d_thresh=6, # coincidence detector threshold
        perm_thr=1, # receptor permanence threshold
        perm_inc=1, # receptor permanence increment
        perm_dec=0) # receptor permanence decrement

    sl.input.add_child(e.output)

    values = [
        'a', 'z', 'a', 'z', 'a', 'z',
        'b', 'z', 'b', 'z', 'b', 'z',
        'c', 'z', 'c', 'z', 'c', 'z',
        'd', 'z', 'd', 'z', 'd', 'z',
        'e', 'z', 'e', 'z', 'e', 'z',
        'f', 'z', 'f', 'z', 'f', 'z',
        'g', 'z', 'g', 'z', 'g', 'z',
        'h', 'z', 'h', 'z', 'h', 'z',
        'i', 'z', 'i', 'z', 'i', 'z',
        'j', 'z', 'j', 'z', 'j', 'z',
        'k', 'z', 'k', 'z', 'k', 'z',
        'l', 'z', 'l', 'z', 'l', 'z',
        'm', 'z', 'm', 'z', 'm', 'z',
        'n', 'z', 'n', 'z', 'n', 'z',
        'o', 'z', 'o', 'z', 'o', 'z',
        'p', 'z', 'p', 'z', 'p', 'z',
        'q', 'z', 'q', 'z', 'q', 'z',
        'r', 'z', 'r', 'z', 'r', 'z',
        's', 'z', 's', 'z', 's', 'z',
        't', 'z', 't', 'z', 't', 'z',
        'u', 'z', 'u', 'z', 'u', 'z',
        'v', 'z', 'v', 'z', 'v', 'z',
        'w', 'z', 'w', 'z', 'w', 'z',
        'x', 'z', 'x', 'z', 'x', 'z',
        'y', 'z', 'y', 'z', 'y', 'z']

    le = preprocessing.LabelEncoder()
    le.fit(values)
    int_values = le.transform(values)

    scores = [0.0 for _ in range(len(values))]
    count_s_output_acts = [0 for _ in range(len(values))]
    count_s_hidden_acts = [0 for _ in range(len(values))]
    count_s_hist = [0 for _ in range(len(values))]
    count_cs = [0 for _ in range(len(values))]
    hidden_s_usage = [0 for _ in range(2240)]
    output_s_usage = [0 for _ in range(2240)]

    print('val  scr  s_act  s_his    cs  active output statelets')

    for i in range(len(int_values)):
        e.compute(value=int_values[i])
        sl.compute(learn=True)

        # update information
        hidden_s_bits = sl.hidden.bits
        hidden_s_acts = sl.hidden.acts
        output_s_bits = sl.output.bits
        output_s_acts = sl.output.acts
        scores[i] = sl.get_score()
        count_s_output_acts[i] = len(output_s_acts)
        count_s_hidden_acts[i] = len(hidden_s_acts)
        count_s_hist[i] = sl.get_historical_count()
        count_cs[i] = sl.get_coincidence_set_count()

        # update statelet usage
        for s in range(len(output_s_usage)):
            hidden_s_usage[s] += hidden_s_bits[s]
            output_s_usage[s] += output_s_bits[s]

        # plot statelets
        if statelet_snapshots_on and (i+1) % 6 == 0:
            title = values[i] + '_' + values[i-1]
            plot_statelets(directory, 'hidden_'+title, hidden_s_bits)
            plot_statelets(directory, 'output_'+title, output_s_bits)

        # print information
        output_s_acts_str = '[' + ', '.join(str(act).rjust(4) for act in output_s_acts) + ']'
        print('{0:>3}  {1:0.1f}  {2:5d}  {3:5d}  {4:4d}  {5:>4}'.format(
            values[i], scores[i], count_s_output_acts[i], count_s_hist[i], count_cs[i], output_s_acts_str))

    # plot information
    plot_results(directory, 'results', values, scores, count_s_output_acts, count_s_hidden_acts, count_s_hist, count_cs, 400, 400)
    plot_statelet_usage(directory, 'hidden', hidden_s_usage, 75)
    plot_statelet_usage(directory, 'output', output_s_usage, 75)

    '''
    # ========================================
    # FOR JACOB
    # ========================================
    print('historical statelets =')
    print(sl.get_historical_statelets())
    print()
    print('num coincidence sets per statelet =')
    print(sl.get_num_coincidence_sets_per_statelet())
    print()
    # ========================================
    # FOR JACOB
    # ========================================
    '''

# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    one_event()
    two_events(statelet_snapshots_on=False) # default
    three_events(statelet_snapshots_on=True)
    multiple_prior_contexts()