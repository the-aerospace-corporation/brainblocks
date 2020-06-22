import numpy as np
from scipy.stats import ortho_group
from sklearn.utils.validation import check_array

np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=200,
                    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})


def print_binary(states):
    for col_index in range(states.shape[0]):
        val_str = "".join(["X" if x else "-" for x in states[col_index, :]])
        print(val_str)

class HyperGridTransform:

    def __init__(self, num_bins=4, num_acts=1, num_grids=100, num_subspace_dims=1, origin=None,
                 max_period=2.0, min_period=0.05, num_input_dims=None,
                 use_orthogonal_bases=False, use_normal_dist_bases=False, use_standard_bases=False,
                 set_bases=None, set_periods=None, use_random_uniform_periods=False,
                 use_evenly_spaced_periods=False,
                 flatten_output=False, random_state=None):

        # check max and min period constraints
        if max_period <= min_period:
            raise Exception("Max period must be greater than min period")

        self.max_period = max_period
        self.min_period = min_period

        # verify custom periods shape
        if set_periods is not None:
            # subspace_periods must be (num_grids, num_subspace_dims)
            custom_shape = set_periods.shape
            if [custom_shape[0] == num_grids, custom_shape[1] == num_subspace_dims].count(False) > 0:
                raise ("custom periods is wrong shape, should be", (num_grids, num_subspace_dims))

        self.set_periods = set_periods

        # random vector basis method
        if [use_normal_dist_bases, use_orthogonal_bases, use_standard_bases, set_bases is not None].count(True) > 1:
            raise Exception("Must choose only one option for subspace vector bases")

        if [use_normal_dist_bases, use_orthogonal_bases, use_standard_bases, set_bases is not None].count(True) == 0:
            use_normal_dist_bases = True

        self.use_normal_dist_bases = use_normal_dist_bases
        self.use_orthogonal_bases = use_orthogonal_bases
        self.use_standard_bases = use_standard_bases
        self.set_bases = set_bases


        # whether to flatten returned binary matrix or preserve multi-dimensionality
        self.flatten_output = flatten_output

        # number of bins
        self.num_bins = num_bins

        # number of active bits per grid
        self.num_acts = num_acts

        # number of vectors m
        self.num_grids = num_grids
        self.num_subspace_dims = num_subspace_dims

        # the total number of bits in output page binary array
        self.num_bits = self.num_grids * (self.num_bins ** self.num_subspace_dims)

        # the total number of active bits in output page binary array
        self.num_act_bits = self.num_grids * (self.num_acts ** self.num_subspace_dims)

        self.use_random_uniform_periods = use_random_uniform_periods
        self.use_evenly_spaced_periods = use_evenly_spaced_periods
        self.flatten_output = flatten_output
        self.random_state = random_state


        # if input dimension not given, set it to subspace dimension
        if num_input_dims is None:
            self.num_features = self.num_subspace_dims
        else:
            self.num_features = num_input_dims

        # don't set origin unless it is given
        if origin is None:
            self.origin = None
        else:
            self.origin = np.array(origin)

        self.fit(np.array([[0.0 for i in range(self.num_features)]]))

    def transform(self, X):
        """Encode scalar features into distributed binary representation
        with hypergrid transform.

        Parameters
        ----------
        X : array of shape (num_samples, num_features)
            Data to be transformed with hypergrids
        Returns
        -------
        X_new : array, shape (num_samples, num_features_new)
            Transformed data
        """
        X = check_array(X)

        X_new = np.apply_along_axis(self._input, 1, X)

        return X_new

    def fit(self, X):
        """
        Fit the HyperGrid to dimensionality of the input, the number of features
        Parameters
        ----------
        X : array-like, shape (num_samples, num_features)
            The data.
        Returns
        -------
        self : instance
        """
        num_samples, num_features = check_array(X, accept_sparse=True).shape

        # number of input features, dimensionality of input space
        self.num_features = num_features


        # check configuration constraints
        if self.num_subspace_dims > self.num_features:
            raise Exception(
                "Target subspace dimension must be less than input space dimension.  Received input %d projected into %d" % (
                    self.num_features, self.num_subspace_dims))

        # origin vector, all vectors are displaced from this point
        if self.origin is None:
            self.origin = tuple([0.0 for k in range(self.num_features)])
        else:
            # if origin is wrong dimension, reset it to origin of correct dimension
            if len(self.origin) != self.num_features:
                new_origin = [0.0 for k in range(self.num_features)]
                for k in range(len(self.origin)):
                    new_origin[k] = self.origin[k]
                self.origin = tuple(new_origin)

        self.origin = np.array(self.origin)

        # verify custom bases shape
        if self.set_bases is not None:
            # subspace_vectors: (num_grids, num_subspace_dims, num_features)
            custom_shape = self.set_bases.shape
            if [custom_shape[0] == self.num_grids, custom_shape[1] == self.num_subspace_dims,
                custom_shape[2] == self.num_features].count(False) > 0:
                raise ("custom bases is wrong shape, should be", (self.num_grids, self.num_subspace_dims, self.num_features))


        # subspace_periods must be (self.num_grids, self.num_subspace_dims)
        if self.set_periods is not None:
            # Custom Magnitudes

            ###########
            self.subspace_periods = self.set_periods

            # set_bases=None, set_periods=None, use_random_uniform_periods=False,
            # use_evenly_spaced_periods=False,
        elif self.use_random_uniform_periods:
            # Random Magnitudes
            # random periods for subspace basis vectors
            ###########
            self.subspace_periods = np.random.uniform(low=self.min_period, high=self.max_period,
                                                      size=self.num_grids * self.num_subspace_dims) \
                .reshape(self.num_grids, self.num_subspace_dims)

        else:
            self.subspace_periods = np.linspace(self.min_period, self.max_period, endpoint=False,
                                                num=self.num_grids * self.num_subspace_dims + 1)[1:]
            self.subspace_periods = self.subspace_periods.reshape(self.num_grids, self.num_subspace_dims)

        if self.use_orthogonal_bases:

            # Random Orthonormal Basis Vectors
            # first num_subspace_dims vectors become our basis of each subspace
            ###########

            self.random_orthonormal_vectors = ortho_group.rvs(self.num_features, self.num_grids)[:, :self.num_subspace_dims, :]

            self.subspace_vectors = self.random_orthonormal_vectors

        elif self.use_normal_dist_bases:

            # Random Normal Basis Vectors from Normal Distribution
            ###########

            # initialize random seed
            self.rand_state = np.random.RandomState(self.random_state)

            # sample from normal distribution that approximates a n-d sphere
            sphere_vectors = self.rand_state.normal(size=(self.num_grids, self.num_subspace_dims, self.num_features))

            # normalize vectors
            sphere_periods = np.linalg.norm(sphere_vectors, axis=2)
            self.random_normal_vectors = sphere_vectors / sphere_periods[..., np.newaxis]

            self.subspace_vectors = self.random_normal_vectors

        elif self.use_standard_bases:
            # Standard Basis Vectors
            ###########

            # random standard basis for each subspace
            # random_standard_basis = np.random.permutation(np.identity(self.num_features))[:num_subspace_dims, :]

            """
            self.random_standard_bases = np.array([
               np.random.permutation(np.identity(self.num_features))[:num_subspace_dims, :]
               for k in range(self.num_grids)
            ])
            """

            self.random_standard_bases = np.array([
                np.random.permutation(
                    np.multiply(np.identity(self.num_features), np.random.choice((1, -1), self.num_features))
                )
                [:self.num_subspace_dims, :]
                for k in range(self.num_grids)
            ])

            # self.subspace_vectors = self.subspace_periods[..., np.newaxis] * self.random_standard_bases
            self.subspace_vectors = self.random_standard_bases

        elif self.set_bases is not None:
            # subspace_vectors: (num_grids, num_subspace_dims, num_features)
            self.subspace_vectors = self.set_bases

        else:
            raise Exception("No subspace vector bases selected")



        return self

    # Fit to data, then transform it.
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : numpy array of shape [num_samples, num_features]
            Training set.
        Returns
        -------
        X_new : numpy array of shape [num_samples, num_features_new]
            Transformed array.
        """

        # fit method of arity 1 (unsupervised transformation)
        return self.fit(X).transform(X)

    def _input(self, input_vec):

        # input_vec: (num_features,)
        # input_vec: (3,)

        # subspace_vectors: (num_grids, num_subspace_dims, num_features)
        # subspace_vectors: (4, 2, 3)

        # input vector in coordinate frame defined by origin
        # disp_vec: (num_features, 1)
        # disp_vec: (3, 1)
        disp_vec = input_vec - self.origin
        disp_vec = np.reshape(disp_vec, (-1, 1))

        # matrix multiply @ to get projection to subspaces
        # proj_vec: (num_grids, num_subspace_dims, 1)
        # proj_vec: (4, 2, 1)
        proj_vec = np.matmul(self.subspace_vectors, disp_vec)

        # grid_periods: (num_grids, num_subspace_dims, 1)
        # grid_periods: (4, 2, 1)
        grid_periods = self.subspace_periods
        grid_periods = grid_periods[..., np.newaxis]

        # origin grid center offset
        # mod_vec: (num_grids, num_subspace_dims, 1)
        # mod_vec: (4, 2, 1)
        mod_vec = np.add(proj_vec, grid_periods / (2.0 * self.num_bins))

        # modulo by period to get value within interval
        # np.fmod creates neg remainder, for neg input
        # np.mod forces pos remainder, for neg input
        mod_vec = np.mod(mod_vec, grid_periods)

        # convert to unit interval
        mod_vec = np.divide(mod_vec, grid_periods)

        # indexes for setting bits
        i_bin = np.floor(mod_vec * self.num_bins).astype(np.int).reshape(self.num_grids, self.num_subspace_dims)
        index1 = i_bin

        # subspace grid shape of m-dimensions with num_acts bits per dimension
        subspace_grid_shape = tuple(self.num_bins for k in range(self.num_subspace_dims))

        # initialize all the bits to zero for num_grids subspaces
        bits_tensor = np.zeros((self.num_grids, *subspace_grid_shape), dtype=np.bool)

        # set the bits for each subspace grid
        for grid_index in range(self.num_grids):
            # activation region of subspace grid
            subspace_acts_shape = tuple(self.num_acts for k in range(self.num_subspace_dims))
            subspace_grid_indices = np.indices(subspace_acts_shape)

            # reshape of index offset matrix
            reshape_list = [-1, ] + [1 for k in range(self.num_subspace_dims)]
            reshape_tuple = tuple(reshape_list)

            # offset activation region of subspace grid
            offset_index = index1[grid_index]
            subspace_grid_indices = subspace_grid_indices + offset_index.reshape(reshape_tuple)

            # modulo the indices to wrap around and prevent ouf range indexes
            subspace_grid_indices = np.mod(subspace_grid_indices, self.num_bins)

            # set the activation region
            bits_tensor[grid_index][tuple(subspace_grid_indices)] = 1

        if self.flatten_output:
            flattened_bits = np.ravel(bits_tensor)
            return flattened_bits
        else:
            return bits_tensor


if __name__ == "__main__":

    # num_features = 3
    # num_grids = 2
    # num_subspace_dims = 2
    """
    custom_bases = np.array([
        [
            [1., 0., 0.],
            [0., 1., 0.]
        ],
        [
            [0., 1., 0.],
            [0., 0., 1.]
        ]
    ])

    custom_periods = np.array([
        [1.0, 2.0],
        [2.0, 1.5]
    ])
    """

    # num_features = 1
    # num_grids = 1
    # num_subspace_dims = 1
    #custom_bases = np.array([
    #    [
    #        [1.]
    #    ]
    #])
    custom_bases = np.array([
       [
           [1.]
       ],
       [
           [1.]
       ]
    ])

    print(custom_bases.shape)

    custom_periods = np.array([
        [1.0],
        [1.1]
     ])
    #custom_periods = np.array([
    #    [1.0]
    #])

    # subspace_vector_bases shape: (num_grids, num_subspace_dims, num_features)
    num_grids = custom_bases.shape[0]
    num_subspace_dims = custom_bases.shape[1]
    num_features = custom_bases.shape[2]

    # configs = {"num_features": 200, "num_grids": 100, "num_bins": 16, "num_acts": 4}
    # configs = {"num_features": 200, "num_grids": 100, "num_bins": 8, "num_acts": 2, "num_subspace_dims": 6,
    #           "flatten_output": True, "max_period": 2.0, "min_period": 0.05, "use_normal_dist_bases": False}
    configs = {"num_grids": num_grids, "num_bins": 4, "num_acts": 1,
               "num_subspace_dims": num_subspace_dims,
               "flatten_output": True, "max_period": 10.0, "min_period": 0.05,
               "use_normal_dist_bases": False,
               "use_standard_bases": False,
               "use_orthogonal_bases": False,
               "set_bases": custom_bases,
               "set_periods": custom_periods
               }
    # configs["origin"] = tuple([random.uniform(-1.0, 1.0) for k in range(configs["num_features"])])
    #configs["origin"] = tuple([0.0 for k in range(configs["num_features"])])

    # instantiate Grid Encoder with initial values
    gridEncoder = HyperGridTransform(**configs)

    # np.random.uniform(low=min_period, high=max_period

    # input vector
    inputs = []
    # for index_value in range(-25, 25):
    for index_value in range(1, 18):
        input_vec = [0.0 for i in range(num_features)]
        input_vec[0] = 0.1 * index_value
        input_vec = np.array(input_vec)
        inputs.append(input_vec)
    inputs = np.array(inputs)

    prev_state = None
    prevInput = None

    gridEncoder = gridEncoder.fit(inputs)

    # get origin values
    origin_vec = np.array([0.0 for i in range(num_features)])
    origin_state = gridEncoder._input(origin_vec).reshape(configs["num_grids"], -1)
    print(origin_state)

    print("all inputs:", inputs)
    states = gridEncoder.transform(inputs)
    print("all outputs:", states)

