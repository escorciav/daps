import numpy as np


def format(X, mthd='c2b'):
    """Transform between temporal/frame annotations

    Parameters
    ----------
    X : ndarray
        2d-ndarray of size [n, 2] with temporal annotations
    mthd : str
        Type of conversion:
        'c2b': transform [center, duration] onto [f-init, f-end]
        'b2c': inverse of c2b
        'd2b': transform ['f-init', 'n-frames'] into ['f-init', 'f-end']

    Outputs
    -------
    Y : ndarray
        2d-ndarray of size [n, 2] with transformed temporal annotations.

    """
    if X.ndim != 2:
        msg = 'Incorrect number of dimensions. X.shape = {}'
        ValueError(msg.format(X.shape))

    if mthd == 'c2b':
        Xinit = np.ceil(X[:, 0] - 0.5*X[:, 1])
        Xend = Xinit + X[:, 1] - 1.0
        return np.stack([Xinit, Xend], axis=-1)
    elif mthd == 'b2c':
        Xc = np.round(0.5*(X[:, 0] + X[:, 1]))
        d = X[:, 1] - X[:, 0] + 1.0
        return np.stack([Xc, d], axis=-1)
    elif mthd == 'd2b':
        Xinit = X[:, 0]
        Xend = X[:, 0] + X[:, 1] - 1.0
        return np.stack([Xinit, Xend], axis=-1)


def intersection(target_segments, test_segments, return_ratio_target=False):
    """Compute intersection btw segments

    Parameters
    ----------
    target_segments : ndarray.
        2d-ndarray of size [m, 2]. The annotation format is [f-init, f-end].
    test_segments : ndarray.
        2d-ndarray of size [m, 2]. The annotation format is [f-init, f-end].
    return_ratio_target : bool, optional.
        extra ndarray output with ratio btw size of intersection over size of
        target-segments.

    Outputs
    -------
    intersect : ndarray.
        3d-ndarray of size [m, n, 2]. The annotation format is [f-init, f-end].
    ratio_target : ndarray.
        2d-ndarray of size [m, n]. Value (i, j) denotes ratio btw size of
        intersect over size of target segment.

    Raises
    ------
    ValueError
        target_segments or test_segments are not 2d array.

    Notes
    -----
    It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')
    m, n = target_segments.shape[0], test_segments.shape[0]
    if return_ratio_target:
        ratio_target = np.zeros((m, n))

    intersect = np.zeros((m, n, 2))
    for i in xrange(m):
        target_size = target_segments[i, 1] - target_segments[i, 0] + 1.0
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        intersect[i, :, 0], intersect[i, :, 1] = tt1, tt2
        if return_ratio_target:
            isegs_size = (tt2 - tt1 + 1.0).clip(0)
            ratio_target[i, :] = isegs_size / target_size

    if return_ratio_target:
        return intersect, ratio_target
    return intersect


def iou(target_segments, test_segments):
    """Compute intersection over union btw segments

    Parameters
    ----------
    target_segments : ndarray.
        2d-ndarray of size [m, 2] with format [t-init, t-end].
    test_segments : ndarray.
        2d-ndarray of size [n x 2] with format [t-init, t-end].

    Outputs
    -------
    iou : ndarray
        2d-ndarray of size [m x n] with tIoU ratio.

    Raises
    ------
    ValueError
        target_segments or test_segments are not 2d-ndarray.

    Notes
    -----
    It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    iou = np.empty((m, n))
    for i in xrange(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        iou[i, :] = intersection / union
    return iou


def non_maxima_supression(dets, score=None, overlap=0.7, measure='iou'):
    """Non-maximum suppression

    Greedily select high-scoring detections and skip detections that are
    significantly covered by a previously selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets : ndarray.
        2d-ndarray of size [num-segments, 2]. Each row is ['f-init', 'f-end'].
    score : ndarray.
        1d-ndarray of with detection scores. Size [num-segments, 2].
    overlap : float, optional.
        Minimum overlap ratio.
    measure : str, optional.
        Overlap measure used to perform NMS either IoU ('iou') or ratio of
        intersection ('overlap')

    Outputs
    -------
    dets : ndarray.
        Remaining after suppression.
    score : ndarray.
        Remaining after suppression.

    Raises
    ------
    ValueError
        - Mismatch between score 1d-array and dets 2d-array
        - Unknown measure for defining overlap

    """
    measure = measure.lower()
    if score is None:
        score = dets[:, 1]
    if score.shape[0] != dets.shape[0]:
        raise ValueError('Mismatch between dets and score.')
    if dets.dtype.kind == "i":
            dets = dets.astype("float")
    # Discard incorrect segments (avoid infinite loop due to NaN)
    idx_correct = np.where(dets[:, 1] > dets[:, 0])[0]

    # Grab coordinates
    t1 = dets[idx_correct, 0]
    t2 = dets[idx_correct, 1]

    area = t2 - t1 + 1
    idx = np.argsort(score[idx_correct])
    pick = []
    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)

        tt1 = np.maximum(t1[i], t1[idx])
        tt2 = np.minimum(t2[i], t2[idx])

        wh = np.maximum(0, tt2 - tt1 + 1)
        if measure == 'overlap':
            o = wh / area[idx]
        elif measure == 'iou':
            o = wh / (area[i] + area[idx] - wh)
        else:
            raise ValueError('Unknown overlap measure for NMS')

        idx = np.delete(idx, np.where(o > overlap)[0])

    return dets[idx_correct[pick], :].astype("int"), score[idx_correct[pick]]
