# ==============================================================================
# bbclassifier.py
# ==============================================================================
import numpy as np
import time
import warnings

from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import DataConversionWarning

from brainblocks.blocks import PatternClassifier, BlankBlock
from brainblocks.tools import HyperGridTransform

class BBClassifier:
    def __init__(self,
                 # Training Arguments
                 num_epochs=3,
                 use_undefined_class=False,

                 # Distributed Pattern Classifier Arguments
                 num_l=2,       # number of labels
                 num_s=512,     # number of statelets
                 num_as=8,      # number of active statelets
                 pct_pool=0.8,  # percent pooled
                 pct_conn=0.8,  # percent initially connected
                 pct_learn=0.3, # percent learn
                 seed=0,

                 # HyperGrid Transform Arguments
                 num_bins=4,
                 num_acts=1,
                 num_grids=64,
                 num_subspace_dims=1,
                 origin=None,
                 num_input_dims=None,
                 max_period=2.0,
                 min_period=0.05,
                 use_orthogonal_bases=False,
                 use_normal_dist_bases=False,
                 use_standard_bases=False,
                 set_bases=None,
                 set_periods=None,
                 use_random_uniform_periods=False,
                 use_evenly_spaced_periods=False,
                 random_state=None):

        """Classify N-dimensional inputs using Hypergrid Transform and Distributed Pattern Classifier

        Parameters
        ----------
        num_epochs: integer
        Number of training epochs

        use_undefined_class: boolean
        Whether to reserve a class for test samples that have no training data

        num_s: integer
        Number of class detectors to allocate for the distributed pattern classifier

        num_as: integer
        Number of active class detectors per time step for the distributed pattern classifier

        pct_pool: float, Between 0.0 and 1.0
        Percentage of random bits an individual detector has potential access to.

        pct_conn: float, Between 0.0 and pct_pool
        Percentage of random bits an individual detector is currently connected to.

        pct_learn: float, Between 0.0 and 1.0
        Percentage of bits to update when training occurs.

        num_bins: integer
        Number of bins to create for each grid.

        num_acts: integer
        Number of contiguous bins to activate along each subspace dimension for each grid.

        num_grids: integer
        Number of grids to generate.

        num_subspace_dims: integer
        Dimensionality of subspaces to map input to

        origin: array-like, shape {num_features}
        Point of origin in input space for embedding a sample into the grids.

        max_period: float
        Maximum bound on grid period

        min_period: float
        Minimum bound on grid period

        use_orthogonal_bases: boolean
        Generate random orthogonal basis vectors for each subspace

        use_normal_dist_bases: boolean
        Generate normal distribution of basis vectors, points sampled on a sphere

        use_standard_bases: boolean
        Use randomly selected standard basis vectors for each grid

        set_bases: array-like, shape {num_grids, num_subspace_dims, num_features}
        Use manually specified subspace basis vectors for each grid

        set_periods: array-like, shape {num_grids, num_subspace_dims}
        Use manually specified periods for each grid and its subspaces

        use_random_uniform_periods: boolean
        Use random periods for subspace grids

        use_evenly_spaced_periods: boolean
        Use evenly spaced periods for subspace grids over the interval min_period to max_period

        random_state: integer
        Seed for random number generators


        """

        self.num_epochs = num_epochs
        self.use_undefined_class = use_undefined_class
        self._y = []
        self.classes_ = np.array([])
        self.outputs_2d_ = False

        self.dpc_config = dict(
            num_l=num_l, num_s=num_s, num_as=num_as, perm_thr=20, perm_inc=2,
            perm_dec=1, pct_pool=pct_pool, pct_conn=pct_conn,
            pct_learn=pct_learn, num_t=2, seed=seed)

        self.hgt_config = dict(
            num_grids=num_grids, num_bins=num_bins, num_acts=num_acts,
            num_subspace_dims=num_subspace_dims, origin=origin,
            num_input_dims=num_input_dims, max_period=max_period,
            min_period=min_period, use_normal_dist_bases=use_normal_dist_bases,
            use_standard_bases=use_standard_bases,
            use_orthogonal_bases=use_orthogonal_bases,
            use_evenly_spaced_periods=use_evenly_spaced_periods,
            use_random_uniform_periods=use_random_uniform_periods,
            set_bases=set_bases, set_periods=set_periods,
            random_state=random_state, flatten_output=True)

    def __del__(self):
        pass

    def _generate_config(self, num_labels=2):
        # connect BlankBlock to DPC
        # optionally add PatternPooler
        pass

    def reset(self):
        pass

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : {array-like, sparse matrix}
            Training data. If array or matrix, shape [num_samples, num_features],

        y : {array-like, sparse matrix}
            Target values of shape = [num_samples] or [num_samples, num_outputs]


        Returns
        -------
        y_new : array, shape (num_samples, num_outputs)
            Classified data
        """

        X, y = check_X_y(X, y, multi_output=True)
        #X, y = check_X_y(X, y)

        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            if y.ndim != 1:
                warnings.warn("A column-vector y was passed when a 1d array "
                              "was expected. Please change the shape of y to "
                              "(num_samples, ), for example using ravel().",
                              DataConversionWarning, stacklevel=2)

            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        check_classification_targets(y)
        self.classes_ = []
        self.class_indices_ = []
        self._y = np.empty(y.shape, dtype=np.int)

        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        #class_names, self.class_indices_ = np.unique(self.classes_, return_inverse=True)

        #self.labels = np.array([str(val) for val in self.class_indices_])

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

        # class for undefined and unseen values
        if self.use_undefined_class:
            undef_class = np.max(self.classes_) + 1
            self.classes_ = np.append(self.classes_, undef_class)
            #print("with undefined class:", self.classes_)

        # update labels in DPC config
        self.dpc_config['num_l'] = len(self.classes_)

        #print("classes_:", self.classes_)
        #print("self._y:", self._y)

        # instantiate the HyperGrid Transform
        self.gridEncoder = HyperGridTransform(**self.hgt_config)

        # fit the HyperGrid transform to the data
        X_new = self.gridEncoder.fit_transform(X)

        #print("HyperGrid Parameters")
        #print(self.gridEncoder.subspace_periods)
        #print(self.gridEncoder.subspace_vectors)

        # get the number of bits being used for transformed output
        self.num_bits = self.gridEncoder.num_bits
        self.num_act_bits = self.gridEncoder.num_act_bits

        #print("BlankBlock:")
        #print("num_bits:", self.num_bits)
        #print("num_act_bits:", self.num_act_bits)

        # Blank Block to hold the hypergrid output
        self.blankBlock = BlankBlock(num_s=self.num_bits)

        # Create PatternClassifier block
        self.dpc = PatternClassifier(**self.dpc_config)

        #print("PatternClassifier:")
        #self.dpc.print_parameters()

        # connect blocks together
        self.dpc.input.add_child(self.blankBlock.output, 0)

        # Train Network
        probs = self._fit(X_new, self._y)
        #print("data:", X_new)
        #print("labels:", self._y)
        #print("training:", probs)

        return self

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (num_query, num_features)
            Test samples.

        Returns
        -------
        y : array of shape [num_samples] or [num_samples, num_outputs]
            Class labels for each data sample.
        """
        X = check_array(X)

        num_samples, num_features = X.shape

        if self._y.ndim == 1 or self._y.ndim == 2 and self._y.shape[1] == 1:
            num_outputs = 1
        else:
            num_outputs = self._y.shape[1]

        # transform data
        X_new = self.gridEncoder.transform(X)

        probabilities = self._predict(X_new)

        classes_ = self.classes_
        if not self.outputs_2d_:
            classes_ = [self.classes_]

        #classes_ = self.labels
        #if not self.outputs_2d_:
        #    classes_ = [self.labels]

        y_pred = np.empty((num_samples, num_outputs), dtype=classes_[0].dtype)

        # print("y_pred:", type(y_pred), y_pred.shape)
        # print("y_pred:", y_pred)

        for k in range(num_samples):
            py = probabilities[k, :]
            y_best = np.argmax(py)
            # print("y_best =", y_best)
            y_pred[k, :] = y_best

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        # print("y_pred:", type(y_pred), y_pred.shape)
        # print("y_pred:", y_pred)

        return y_pred



    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (num_query, num_features), \
                or (num_query, num_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : array of shape = [num_samples, num_classes], or a list of num_outputs
            of such arrays if num_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = check_array(X)

        num_samples, num_features = X.shape

        if self._y.ndim == 1 or self._y.ndim == 2 and self._y.shape[1] == 1:
            num_outputs = 1
        else:
            num_outputs = self._y.shape[1]

        # transform data
        X_new = self.gridEncoder.transform(X)

        return self._predict(X_new)

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (num_samples, num_features)
            Test samples.

        y : array-like, shape = (num_samples) or (num_samples, num_outputs)
            True labels for X.

        sample_weight : array-like, shape = [num_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def _fit(self, X, y):

        probabilities = []
        # train pattern classifier
        for i in range(self.num_epochs):
            epoch_probs = []

            # Train Network
            #t0 = time.time()
            for k in range(y.shape[0]):
                input = X[k, :]
                target = y[k]

                self.blankBlock.output.bits = input
                self.blankBlock.feedforward()
                self.dpc.set_label(target)
                self.dpc.feedforward(learn=True)

                curr_prob = self.dpc.get_probabilities()
                epoch_probs.append(curr_prob)

            #t1 = time.time()
            #print("train epoch time = %fs with size %d" % ((t1 - t0), y.shape[0]))
            probabilities.append(epoch_probs)

        return np.asarray(probabilities)

    def _predict(self, X):

        probabilities = []

        # num_points = 1000
        num_points = X.shape[0]

        #t0 = time.time()
        for k in range(X.shape[0]):
            input = X[k, :]

            self.blankBlock.output.bits = input
            self.blankBlock.feedforward()
            self.dpc.feedforward(learn=False)

            curr_prob = self.dpc.get_probabilities()
            probabilities.append(curr_prob)

        #t1 = time.time()
        #print("%d points, time = %fs" % (num_points, (t1 - t0)))

        return np.asarray(probabilities)




if __name__ == "__main__":

    TRAINS = np.array([
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0])

    TESTS = np.array([0.0, 1.0])

    LABELS = np.array([
        'a', 'b', 'a', 'b', 'a',
        'b', 'a', 'b', 'a', 'b',
        'a', 'b', 'a', 'b', 'a',
        'b', 'a', 'b', 'a', 'b',
        'a', 'b', 'a', 'b', 'a',
        'b', 'a', 'b', 'a', 'b',
        'a', 'b', 'a', 'b', 'a'])

    TRAINS = TRAINS.reshape(TRAINS.size, 1)
    LABELS = LABELS.reshape(LABELS.size,)
    TESTS = TESTS.reshape(TESTS.size, 1)

    classifier_config = {"num_grids": 64, "num_bins": 4,
               "max_period": 2.0, "min_period": 0.05,
               "use_normal_dist_bases": True,
               "use_evenly_spaced_periods": True,
               #"labels": ["a", "b"]
               }

    # instantiate Grid Encoder with initial values
    bbClassifier = BBClassifier(**classifier_config)

    bbClassifier.fit(TRAINS, LABELS)
    preds = bbClassifier.predict(TESTS)

    print("TESTS:", TESTS)
    print("preds:", preds)

    #print("input=%0.1f, prob(a)=%f, prob(b)=%f" % (TESTS[0], probs[0][0], probs[0][1]))
    #print("input=%0.1f, prob(a)=%f, prob(b)=%f" % (TESTS[1], probs[1][0], probs[1][1]))
