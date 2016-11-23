import numpy as np


def concat1d(x, n=8, pool_type='mean', norm=True, unit=False):
    """1D-concat representation of multiple feature vectorz

    Parameters
    ----------
    x : ndarray.
        [m x d] array of features. m is the number of features and d is the
        dimensionality of the feature space.
    n : int
        Number of chunks.
    pool_type : str, optional.
        Pooling strategy over a bunch of features.
    norm : bool, optional.
        Normalize each region before concatenate them.
    unit : bool, optional.
        Normalize the final input vector.

    Outputs
    -------
    concat_feat : ndarray
        [d * n] ndarray with concat feature of x.

    Raises
    ------
    ValueError
        when n > m.

    """
    m, d = x.shape
    if n > m:
        raise ValueError(
            'n > m. This is an odd case. Define appropriate value for n '
            'considering that num-chunks {} > num-features {}'.format(n, m))
    arr = [np.empty(d) for i in range(n)]
    pool_type = pool_type.lower()

    edges = np.ones(n + 1, dtype=int) * 1.0 / n
    edges[0] = 0
    edges = np.round(np.cumsum(edges) * m).astype(int)
    for j in range(n):
        if pool_type == 'mean':
            arr[j][...] = x[edges[j]:edges[j + 1], :].mean(axis=0)
        elif pool_type == 'max':
            arr[j][...] = x[edges[j]:edges[j + 1], :].max(axis=0)
        else:
            raise ValueError('Unknown pooling type {}'.format(pool_type))

        if norm:
            feat_norm = np.sqrt((arr[j] ** 2).sum())
            if feat_norm == 0:
                feat_norm = 1.0
            arr[j] /= feat_norm

    concat_feat = np.hstack(arr)
    if unit:
        return concat_feat / n
    return concat_feat
