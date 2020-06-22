import numpy as np

def otsuka_similarity(X_gnomes, ref_gnomes):
    """Otsuka similarity score.

    # def otsuka(x, N=10):
    # 		- v1 . v2 / sqrt( |v1|_1 * |v2|_1 )
    # return x / np.sqrt(N*N)

    :param X_gnomes: array_like
        An array with shape (n_samples, n_features).

    :param ref_gnomes: array_like
        An array with shape (n_samples, n_features).

    :return: array
        An array of scores from 0.0 to 1.0 with shape (n_samples)
    """


    # dot product of bit vectors
    dot_product = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))

    l1_norm_1 = np.count_nonzero(X_gnomes, axis=1)

    l1_norm_2 = np.count_nonzero(ref_gnomes, axis=1)

    # sum_of_squares = np.add(square1.reshape(-1, 1), square2.reshape(1, -1))
    product_of_magnitudes = np.multiply(l1_norm_1.reshape(-1, 1), l1_norm_2.reshape(1, -1))

    normalizing_denominator = np.sqrt(product_of_magnitudes)

    normalized_scores = np.divide(dot_product, normalizing_denominator)

    return normalized_scores


def tanimoto_similarity(X_gnomes, ref_gnomes):
    """Tanimoto similarity score.

    # def tanimoto(x, N=10):
    # 		- v1 . v2 / ( |v1|^2 + |v2|^2 - v1 . v2)
    # return x / ( N + N - x)

    :param X_gnomes: array_like
        An array with shape (n_samples, n_features).

    :param ref_gnomes: array_like
        An array with shape (n_samples, n_features).

    :return: array
        An array of scores from 0.0 to 1.0 with shape (n_samples)
    """

    # dot product of bit vectors
    dot_product = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))

    l2_norm_1 = np.linalg.norm(X_gnomes.astype(int), axis=1)
    l2_norm_2 = np.linalg.norm(ref_gnomes.astype(int), axis=1)

    square1 = np.square(l2_norm_1)
    square2 = np.square(l2_norm_2)

    sum_of_squares = np.add(square1.reshape(-1, 1), square2.reshape(1, -1))

    normalizing_denominator = sum_of_squares - dot_product

    normalized_scores = np.divide(dot_product, normalizing_denominator)

    return normalized_scores


def weighted_similarity(X_gnomes, ref_gnomes):
    """Weighted similarity score.

    :param X_gnomes: array_like
        An array with shape (n_samples, n_features).

    :param ref_gnomes: array_like
        An array with shape (n_samples, n_features).

    :return: array
        An array of scores from 0.0 to 1.0 with shape (n_samples)
    """

    dot_product = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))

    l1_norm_1 = np.count_nonzero(X_gnomes, axis=1)
    l1_norm_2 = np.count_nonzero(ref_gnomes, axis=1)

    sum_of_norms = np.add(l1_norm_1.reshape(-1, 1), l1_norm_2.reshape(1, -1))

    half_sum = sum_of_norms / 2

    normalized_scores = np.divide(dot_product, half_sum)

    return normalized_scores


def gnome_similarity(X_gnomes, ref_gnomes):
    """Gnome similarity score.

    Asymmetric score with respect to X_gnomes

    :param X_gnomes: array_like
        An array with shape (n_samples, n_features).

    :param ref_gnomes: array_like
        An array with shape (n_samples, n_features).

    :return: array
        An array of scores from 0.0 to 1.0 with shape (n_samples)

    """
    sum_scores = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))
    l1_norm = np.count_nonzero(X_gnomes, axis=1)
    normalized_scores = np.divide(sum_scores, l1_norm[:, np.newaxis])

    return normalized_scores