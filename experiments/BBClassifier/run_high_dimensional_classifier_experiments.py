#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from brainblocks.tools import BBClassifier
from brainblocks.datasets.classification import make_box_data_random, make_box_data_grid
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs, make_s_curve, make_swiss_roll
import hypertools as hyp
import warnings

from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')


def plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str, use_undefined_class=False):
    color_maps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys']
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'grey']

    # plot results and data
    # figure = plt.figure(figsize=(21, 30), constrained_layout=True)
    figure = plt.figure(figsize=(21, 12), constrained_layout=True)
    i = 1

    x_boundary = 0.5
    y_boundary = 0.5

    for ds_cnt, results in enumerate(results_per_dataset):

        num_datasets = len(results_per_dataset)
        num_classifiers = len(results_per_dataset[0]['results'])

        X, y, X_plot, X_train, X_test, y_train, y_test, X_plot_train, X_plot_test = results['data']

        # max and min of plot boundaries
        x_min, x_max = X_plot[:, 0].min() - x_boundary, X_plot[:, 0].max() + x_boundary
        y_min, y_max = X_plot[:, 1].min() - y_boundary, X_plot[:, 1].max() + y_boundary

        # color map based on number of classes
        n_classes = np.unique(y).size
        # n_classes += 1
        cm_bright = ListedColormap(color_list[:n_classes])

        # just plot the dataset first
        ax = plt.subplot(num_datasets, num_classifiers + 1, i)

        if ds_cnt == 0:
            ax.set_title("Training Data", fontsize=16)

        # print(y_train)
        # print(cm_bright(y_train))

        # Plot the training points
        ax.scatter(X_plot_train[:, 0], X_plot_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        results_per_classifier = results['results']

        n_regions = n_classes

        if use_undefined_class:
            n_regions += 1

        # iterate over classifiers
        for name, score in results_per_classifier:

            ax = plt.subplot(num_datasets, num_classifiers + 1, i)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].

            """
            
            for k in range(n_regions):
                Z_class = Z[:, k].reshape(xx.shape)

                class_colors = get_cmap(color_maps[k % n_regions], 100)
                class_colors = class_colors(np.linspace(0, 1, 100))
                class_colors[:, 3] = np.linspace(0, 1, 100)

                class_cmp = ListedColormap(class_colors)

                ax.contourf(xx, yy, Z_class, cmap=class_cmp)
            """

            # Plot the testing points
            ax.scatter(X_plot_test[:, 0], X_plot_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name, fontsize=16)
            ax.text(x_max - .1, y_min + .1, ('%.2f' % score).lstrip('0'),
                    size=25, horizontalalignment='right')
            i += 1

    figure.suptitle(title_str + "\n", fontsize=25)

    # plt.tight_layout()
    plt.savefig(filename_str)
    plt.close()


def evaluate(datasets, classifier_configs):
    # Evaluation Configuration
    #
    # h = .02
    h = 0.05

    x_boundary = 0.5
    y_boundary = 0.5
    # x_boundary = 6.5
    # y_boundary = 6.5
    # x_boundary = 2.5
    # y_boundary = 2.5
    # j_boundary = 12.0
    # y_boundary = 12.0

    # iterate over datasets
    results_per_dataset = []
    for ds_cnt, ds in enumerate(datasets):

        print("Data Set %d" % ds_cnt)

        # preprocess dataset, split into training and test part
        X, y, X_plot = ds
        X = MinMaxScaler().fit_transform(X)
        X_plot = MinMaxScaler().fit_transform(X_plot)

        # uniform mesh
        # x_min, x_max = X[:, 0].min() - x_boundary, X[:, 0].max() + x_boundary
        # y_min, y_max = X[:, 1].min() - y_boundary, X[:, 1].max() + y_boundary
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                     np.arange(y_min, y_max, h))

        X_train, X_test, y_train, y_test, X_plot_train, X_plot_test = \
            train_test_split(X, y, X_plot, test_size=.1, random_state=42)

        # all data in a tuple for comparison across classifiers and for plotting
        # split_data = (X, y, xx, yy, X_train, X_test, y_train, y_test)
        split_data = (X, y, X_plot, X_train, X_test, y_train, y_test, X_plot_train, X_plot_test)

        # collect results for each classifier experiment on this data
        results_per_classifier = []

        # iterate over classifiers
        for name, bb_config in classifier_configs:
            num_input_dims = X_train.shape[1]
            bb_config['num_input_dims'] = num_input_dims

            # clf = KNeighborsClassifier(3)
            clf = BBClassifier(**bb_config)

            print("Train %s" % name)

            # train the model
            clf.fit(X_train, y_train)

            # compute accuracy on test values
            score = clf.score(X_test, y_test)

            results_per_classifier.append((name, score))

            del clf

        data_result = {'data': split_data, 'results': results_per_classifier}
        results_per_dataset.append(data_result)

    return results_per_dataset


def run_encoder_act_bits_experiment(datasets):
    # FIXME: Unable to test, no mesh implementation available
    raise("Unable to test, no mesh implementation available")
    filename_str = "mesh_encoder_act_bits.png"
    title_str = "Mesh Encoder Active Bit Width Comparision"

    # Classifier Configurations
    #
    classifier_configs = []

    bb_config = {'num_bits': 128, 'num_as': 4}
    clf_name1 = "%d / %d bits" % (bb_config['num_as'], bb_config['num_bits'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_bits': 128, 'num_as': 8}
    clf_name1 = "%d / %d bits" % (bb_config['num_as'], bb_config['num_bits'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_bits': 128, 'num_as': 16}
    clf_name1 = "%d / %d bits" % (bb_config['num_as'], bb_config['num_bits'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_bits': 128, 'num_as': 32}
    clf_name1 = "%d / %d bits" % (bb_config['num_as'], bb_config['num_bits'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_bits': 128, 'num_as': 64}
    clf_name1 = "%d / %d bits" % (bb_config['num_as'], bb_config['num_bits'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_bits': 128, 'num_as': 112}
    clf_name2 = "%d / %d bits" % (bb_config['num_as'], bb_config['num_bits'])
    classifier_configs.append((clf_name2, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_grid_vector_magnitude_experiment(datasets):
    filename_str = "high_dimension_period_experiment.png"
    title_str = "High Dimensional Period Data Experiment"

    # Classifier Configurations
    #
    classifier_configs = []

    X, y, X_plot = datasets[0]

    num_input_dims = X.shape[1]

    print("num_input_dims=", num_input_dims)

    # bb_config = {'use_binary': True, 'num_grids': 10}
    # clf_name1 = 'Grid Vectors %d' % bb_config['num_grids']
    # classifier_configs.append((clf_name1, bb_config))

    # template_bb_config = {'use_grid': True, 'num_s': 300, 'num_as': 10}
    # template_bb_config = {'use_grid': True, 'num_input_dims': 3, 'num_bins': 16, 'num_as': 4,
    #                      'num_grids': 80, 'max_period': 1.9, 'min_period': 1.3}
    template_bb_config = {'num_input_dims': num_input_dims, 'num_bins': 16, 'num_as': 4,
                          'num_grids': 80, 'max_period': 3.0, 'num_s': 1024, 'min_period': 1.0}

    #clf_name1 = 'Dimensions = %d' % num_input_dims
    bb_config = deepcopy(template_bb_config)
    bb_config['max_period'] = 3.0
    bb_config['min_period'] = bb_config['max_period'] - 0.01
    clf_name1 = 'Period %1.1f' % bb_config['max_period']
    classifier_configs.append((clf_name1, bb_config))

    bb_config = deepcopy(template_bb_config)
    bb_config['max_period'] = 2.5
    bb_config['min_period'] = bb_config['max_period'] - 0.01
    clf_name1 = 'Period %1.1f' % bb_config['max_period']
    classifier_configs.append((clf_name1, bb_config))

    bb_config = deepcopy(template_bb_config)
    bb_config['max_period'] = 2.0
    bb_config['min_period'] = bb_config['max_period'] - 0.01
    clf_name1 = 'Period %1.1f' % bb_config['max_period']
    classifier_configs.append((clf_name1, bb_config))

    bb_config = deepcopy(template_bb_config)
    bb_config['max_period'] = 1.6
    bb_config['min_period'] = bb_config['max_period'] - 0.01
    clf_name1 = 'Period %1.1f' % bb_config['max_period']
    classifier_configs.append((clf_name1, bb_config))

    bb_config = deepcopy(template_bb_config)
    bb_config['max_period'] = 1.1
    bb_config['min_period'] = bb_config['max_period'] - 0.01
    clf_name1 = 'Period %1.1f' % bb_config['max_period']
    classifier_configs.append((clf_name1, bb_config))

    bb_config = deepcopy(template_bb_config)
    bb_config['max_period'] = 0.8
    bb_config['min_period'] = bb_config['max_period'] - 0.01
    clf_name1 = 'Period %1.1f' % bb_config['max_period']
    classifier_configs.append((clf_name1, bb_config))

    bb_config = deepcopy(template_bb_config)
    bb_config['max_period'] = 0.5
    bb_config['min_period'] = bb_config['max_period'] - 0.01
    clf_name1 = 'Period %1.1f' % bb_config['max_period']
    classifier_configs.append((clf_name1, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_basic_grid_encoder_experiment(datasets):
    filename_str = "high_dimension_basic_hypergrid_experiment.png"
    title_str = "High Dimensional Basic Hypergrid Data Experiment"

    # Classifier Configurations
    #
    classifier_configs = []

    template_bb_config = {'num_input_dims': 2, 'num_bins': 16, 'num_as': 4,
                          'num_grids': 80, 'max_period': 1.9, 'min_period': 1.3}

    bb_config = deepcopy(template_bb_config)
    bb_config['num_grids'] = 160
    clf_name1 = 'Grids %d' % bb_config['num_grids']
    classifier_configs.append((clf_name1, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_basic_mesh_encoder_experiment(datasets):
    # FIXME: Unable to test, no mesh implementation available
    raise("Unable to test, no mesh implementation available")
    filename_str = "basic_mesh_encoder.png"
    title_str = "Mesh Encoder Training Epochs"

    # Classifier Configurations
    #
    classifier_configs = []

    bb_config = {'num_s': 256, 'num_as': 1}
    # bb_config = {'use_mesh': True, 'num_s': 300, 'num_as': 15}
    clf_name1 = '15/300 neurons'
    classifier_configs.append((clf_name1, bb_config))

    # bb_config = {'use_mesh': True, 'num_s': 300, 'num_as': 15}
    # clf_name1 = '15/300 neurons'
    # classifier_configs.append((clf_name1, bb_config))

    # bb_config = {'use_mesh': True, 'num_s': 400, 'num_as': 20}
    # clf_name1 = '20/400 neurons'
    # classifier_configs.append((clf_name1, bb_config))

    # bb_config = {'use_mesh': True, 'num_s': 500, 'num_as': 25}
    # clf_name1 = '25/500 neurons'
    # classifier_configs.append((clf_name1, bb_config))

    # bb_config = {'use_mesh': True, 'num_s': 300, 'num_as': 15, 'num_epochs': 8}
    # clf_name1 = '8 Epochs'
    # classifier_configs.append((clf_name1, bb_config))

    # bb_config = {'use_mesh': True, 'num_s': 300, 'num_as': 15, 'num_epochs': 16}
    # clf_name1 = '16 Epochs'
    # classifier_configs.append((clf_name1, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # print(results_per_dataset)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_many_neuron_experiment(datasets):
    filename_str = "high_dimension_many_neuron_experiment.png"
    # title_str = "Mesh Encoder Training Epochs"
    title_str = "High Dimensional Many Neuron Training Epochs"

    # Classifier Configurations
    #
    classifier_configs = []

    bb_config = {'num_s': 1024, 'num_as': 1, 'num_epochs': 3}
    clf_name1 = '16/1024 neurons, 3 Epochs'
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_s': 1024, 'num_as': 1, 'num_epochs': 6}
    clf_name1 = '16/1024 neurons, 6 Epochs'
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_s': 1024, 'num_as': 1, 'num_epochs': 9}
    clf_name1 = '16/1024 neurons, 9 Epochs'
    classifier_configs.append((clf_name1, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # print(results_per_dataset)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_mesh_vs_scalar_encoder_experiment(datasets):
    # FIXME: Unable to test, no mesh implementation available
    raise("Unable to test, no mesh implementation available")
    filename_str = "mesh_vs_scalar_encoder.png"
    title_str = "Mesh Encoder vs 2 Scalar Encoders"

    # Classifier Configurations
    #
    classifier_configs = []

    bb_config = {'use_mesh': True}
    clf_name1 = '2D Mesh Encoder'
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'use_mesh': False}
    clf_name2 = '2 Scalar Encoders'
    classifier_configs.append((clf_name2, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_encoder_limit_experiment(datasets):
    # FIXME: Unable to test, no mesh implementation available
    raise("Unable to test, no mesh implementation available")
    filename_str = "mesh_encoder_limit_comparison.png"
    title_str = "Mesh Encoder Limit Comparison"

    # Classifier Configurations
    #
    classifier_configs = []

    bb_config = {'min_val': -0.5, 'max_val': 1.5}
    clf_name1 = "Limit (%0.1f,%0.1f)" % (bb_config['min_val'], bb_config['max_val'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'min_val': 0.0, 'max_val': 1.0}
    clf_name1 = "Limit (%0.1f,%0.1f)" % (bb_config['min_val'], bb_config['max_val'])
    classifier_configs.append((clf_name1, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_classifier_active_percent_experiment(datasets):
    # FIXME: Unable to test since activation rate is not controllable with current implementation
    raise("Unable to test since activation rate is not controllable with current implementation")

    filename_str = "classifier_active_neurons_comparison.png"
    title_str = "Classifier Active Neurons Comparison"

    # Classifier Configurations
    #
    classifier_configs = []

    bb_config = {'num_as': 1, 'pct_pool': 0.1}
    clf_name1 = "num_as %d" % (bb_config['num_as'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_as': 2, 'pct_pool': 0.1}
    clf_name1 = "num_as %d" % (bb_config['num_as'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_as': 3, 'pct_pool': 0.1}
    clf_name1 = "num_as %d" % (bb_config['num_as'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_as': 4, 'pct_pool': 0.1}
    clf_name1 = "num_as %d" % (bb_config['num_as'])
    classifier_configs.append((clf_name1, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_classifier_total_neurons_experiment(datasets):
    filename_str = "high_dimension_classifier_total_neurons_comparison.png"
    title_str = "High Dimensional Classifier Total Neurons Comparison"

    # Classifier Configurations
    #
    classifier_configs = []

    bb_config = {'num_s': 128}
    clf_name1 = "Total Neurons %d" % (bb_config['num_s'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_s': 256}
    clf_name1 = "Total Neurons %d" % (bb_config['num_s'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_s': 512}
    clf_name1 = "Total Neurons %d" % (bb_config['num_s'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'num_s': 1024}
    clf_name1 = "Total Neurons %d" % (bb_config['num_s'])
    classifier_configs.append((clf_name1, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


def run_classifier_synaptic_potential_percent_experiment(datasets):
    filename_str = "high_dimension_classifier_synaptic_potential_percent_comparison.png"
    title_str = "High Dimensional Classifier Synaptic Potential Percent Comparison"

    # Classifier Configurations
    #
    classifier_configs = []

    bb_config = {'pct_pool': 0.1}
    clf_name1 = "Potential Percent %0.1f" % (bb_config['pct_pool'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'pct_pool': 0.2}
    clf_name1 = "Potential Percent %0.1f" % (bb_config['pct_pool'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'pct_pool': 0.4}
    clf_name1 = "Potential Percent %0.1f" % (bb_config['pct_pool'])
    classifier_configs.append((clf_name1, bb_config))

    bb_config = {'pct_pool': 0.8}
    clf_name1 = "Potential Percent %0.1f" % (bb_config['pct_pool'])
    classifier_configs.append((clf_name1, bb_config))

    # evaluate classifiers based on configurations and datasets
    results_per_dataset = evaluate(datasets, classifier_configs)

    # Plot the results of the experiments
    plot_classifier_results_by_dataset(results_per_dataset, title_str, filename_str)


# Grid Experiments

# 1) vary the number of bins, fixed activation (changes total neurons)
# 2) vary the number active bits, fixed number of bins (changes total neurons)
# 3) evaluate various fixed magnitudes for all vectors (0.05 to 3.0 are current mag range)
# 4) evaluate larger contour space to see if there's any aliasing (-4.5,-4.5 to 5.5,5.5), reduce contour resolution
# 5) pick two fixed magnitudes, and vary their distribution ratio
# 6) identify which vector neurons and magnitudes are used for identifying the classes
# 7) Do 3 dimensional data
# 8) configure individual vector projection encoders with number of bins, number of active bits, degree of overlap


if __name__ == "__main__":
    # Datasets
    #
    noise = 0.1
    random_state = 170
    cluster_std = 4.0
    n_samples = 200

    # dimensionality of data
    n_features = 20
    # n_features = 14
    # n_features = 3
    # n_reduced_features = 4
    reduction_method = 'UMAP'

    # Regression Datasets

    # make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1, bias=0.0,
    # effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)

    # Classification Datasets

    # MNIST
    # Imagenet

    # load_digits(n_class=10, return_X_y=False)
    # Classes	10
    # Samples per class	~180
    # Samples total	1797
    # Dimensionality	64
    # Features	integers 0-16

    # load_iris(return_X_y=False)
    # Classes	3
    # Samples per class	50
    # Samples total	150
    # Dimensionality	4

    # Manifold learning datasets

    # make_swiss_roll(n_samples=100, noise=0.0, random_state=None)
    # returns
    # X : array of shape [n_samples, 3]
    # t : array of shape [n_samples]

    # make_s_curve(n_samples=100, noise=0.0, random_state=None)
    # returns
    # X : array of shape [n_samples, 3]
    # t : array of shape [n_samples]

    # datasets = []
    # for k in range(20, 30):
    #    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=k, cluster_std=cluster_std,
    #                      random_state=random_state)
    #    X_plot = hyp.reduce(X, ndims=2, reduce=reduction_method)
    #    datasets.append((X, y, X_plot))

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=3, n_redundant=17,
                               n_repeated=0,
                               n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=3.0, hypercube=True,
                               shift=0.0, scale=1.0, shuffle=True, random_state=42)
    X_plot = hyp.reduce(X, ndims=2, reduce=reduction_method)
    data1 = (X, y, X_plot)

    datasets = [data1]

    # ============

    # hyp.plot(X, group=y,  align='hyper')
    # hyp.describe(X, reduce='UMAP', max_dims=5)
    # hyp.describe(X, reduce='UMAP')
    # hyp.plot(X, '.', reduce='UMAP', hue=y, ndims=2)
    # X = hyp.reduce(X_raw, ndims=n_features, reduce=reduction_method)
    # X_reduce = hyp.reduce(X, ndims=n_reduced_features, reduce=reduction_method)
    # X_plot = hyp.reduce(X, ndims=2, reduce=reduction_method)

    # data1 = make_box_data_random(n_samples=num_samples, min_val=-0.3, max_val=1.3, stratify=True, shuffle=True)
    # data2 = make_box_data_grid(h=0.05, min_val=-0.3, max_val=1.3, shuffle=True)
    # datasets = [data1]
    # data1 = make_box_data_random(n_samples=num_samples, min_val=-0.3, max_val=1.3, stratify=True, shuffle=True)
    # data2 = make_box_data_grid(h=0.05, min_val=-0.3, max_val=1.3, shuffle=True)
    # datasets = [data1, data2]

    # data1 = make_blobs(n_samples=num_samples, n_features=2, centers=2)
    # datasets = [data1]


    # run_basic_mesh_encoder_experiment(datasets)
    # run_encoder_act_bits_experiment(datasets)
    # run_mesh_vs_scalar_encoder_experiment(datasets)
    # run_encoder_limit_experiment(datasets)
    # run_classifier_active_percent_experiment(datasets)

    run_classifier_total_neurons_experiment(datasets)
    run_basic_grid_encoder_experiment(datasets)
    run_grid_vector_magnitude_experiment(datasets)
    run_classifier_synaptic_potential_percent_experiment(datasets)

