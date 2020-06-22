import numpy as np




def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    xinters = -1e30

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def make_box_data_grid(h=0.05, max_val=1.1, min_val=-0.1, shuffle=True, rand_seed=2):
    polyRegions = []
    polyRegions.append([(0.1, 0.1), (0.1, 0.4), (0.4, 0.4), (0.4, 0.1)])
    polyRegions.append([(0.6, 0.6), (0.6, 0.9), (0.9, 0.9), (0.9, 0.6)])

    labels = []

    x_min, x_max = min_val, max_val
    y_min, y_max = min_val, max_val
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    xx = xx.ravel()
    yy = yy.ravel()
    zipped_data = list(zip(xx, yy))
    data = np.asarray(zipped_data)
    X = data

    # for k in range(num_samples):
    for k in range(X.shape[0]):
        if point_inside_polygon(X[k, 0], X[k, 1], polyRegions[0]) or point_inside_polygon(X[k, 0], X[k, 1],
                                                                                          polyRegions[1]):
            # inside
            labels.append(0)
        else:
            # outside
            labels.append(1)

    y = np.asarray(labels)

    if shuffle:
        rng = np.random.RandomState(rand_seed)
        rng.shuffle(X)
        rng = np.random.RandomState(rand_seed)
        rng.shuffle(y)

    return X, y


def make_box_data_random(n_samples=500, max_val=1.1, min_val=-0.1, stratify=False, shuffle=True, rand_seed=2):
    polyRegions = []
    polyRegions.append([(0.1, 0.1), (0.1, 0.4), (0.4, 0.4), (0.4, 0.1)])
    polyRegions.append([(0.6, 0.6), (0.6, 0.9), (0.9, 0.9), (0.9, 0.6)])

    labels = []

    # create balanced set of inside/outside samples

    if stratify:
        num_in_samples = n_samples / 2
        num_out_samples = n_samples - num_in_samples

        rng = np.random.RandomState(rand_seed)

        in_samples = []
        out_samples = []

        while len(out_samples) < num_out_samples or len(in_samples) < num_in_samples:
            X = rng.uniform(low=min_val, high=max_val, size=2)

            if point_inside_polygon(X[0], X[1], polyRegions[0]) \
                    or point_inside_polygon(X[0], X[1], polyRegions[1]):
                # inside
                if len(in_samples) < num_in_samples:
                    in_samples.append(X)
            else:
                # outside
                if len(out_samples) < num_out_samples:
                    out_samples.append(X)

        in_labels = [0 for k in range(len(in_samples))]
        out_labels = [1 for k in range(len(out_samples))]
        labels = in_labels + out_labels
        y = np.asarray(labels)

        all_samples = in_samples + out_samples
        X = np.asarray(all_samples)

    else:
        rng = np.random.RandomState(rand_seed)
        X = rng.uniform(low=min_val, high=max_val, size=(n_samples, 2))

        # for k in range(n_samples):
        for k in range(X.shape[0]):
            if point_inside_polygon(X[k, 0], X[k, 1], polyRegions[0]) \
                    or point_inside_polygon(X[k, 0], X[k, 1], polyRegions[1]):
                # inside
                labels.append(0)
            else:
                # outside
                labels.append(1)

        y = np.asarray(labels)

    if shuffle:
        rng = np.random.RandomState(rand_seed)
        rng.shuffle(X)
        rng = np.random.RandomState(rand_seed)
        rng.shuffle(y)

    return X, y


