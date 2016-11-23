import nose.tools as nt
import numpy as np

from daps.utils.pooling import concat1d


def test_concat1d():
    m, d = 5, 2
    a = np.arange(m*d).reshape((m, d))
    nt.assert_raises(ValueError, concat1d, a, m + 1)
    answer = [[1, 2, 6, 7], [2, 3, 8, 9]]
    for i, pool_type in enumerate(['mean', 'max']):
        rst = concat1d(a, 2, pool_type, False, False)
        np.testing.assert_equal(rst, answer[i])
    # TODO: test norm, unit flags
