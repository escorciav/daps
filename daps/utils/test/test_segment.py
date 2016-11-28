import unittest

import numpy as np

import daps.utils.segment as segment


class test_segment_utilities(unittest.TestCase):
    def test_format(self):
        # Test d2b
        boxes = np.array([[10, 3], [5, 7]])
        bsol = np.array([[10, 12], [5, 11]])
        bout = segment.format(boxes, 'd2b')
        np.testing.assert_array_equal(bout, bsol)
        # Test b2c
        boxes = bout.copy()
        bsol = np.array([[11, 3], [8, 7]])
        bout = segment.format(boxes, 'b2c')
        np.testing.assert_array_equal(bout, bsol)
        # Test c2b
        boxes = bout.copy()
        bsol = np.array([[10, 12], [5, 11]])
        bout = segment.format(boxes, 'c2b')
        np.testing.assert_array_equal(bout, bsol)

    def test_intersection(self):
        a = np.random.rand(1)
        b = np.array([[1, 10], [5, 20], [16, 25]])
        self.assertRaises(ValueError, segment.intersection, a, b)
        a = np.random.rand(100, 2)
        self.assertEqual((100, 3, 2), segment.intersection(a, b).shape)
        a = np.array([[5, 15]])
        gt_isegs = np.array([[[5, 10], [5, 15], [16, 15]]], dtype=float)
        np.testing.assert_array_equal(gt_isegs, segment.intersection(a, b))
        results = segment.intersection(a, b, True)
        self.assertEqual(2, len(results))
        self.assertEqual((a.shape[0], b.shape[0]), results[1].shape)

    def test_iou(self):
        a = np.array([[1, 10], [5, 20], [16, 25]])
        b = np.random.rand(1)
        self.assertRaises(ValueError, segment.iou, a, b)
        b = np.random.rand(100, 2)
        self.assertEqual((3, 100), segment.iou(a, b).shape)
        b = np.array([[1, 10], [1, 30], [10, 20], [20, 30]])
        rst = segment.iou(a, b)
        # segment is equal
        self.assertEqual(1.0, rst[0, 0])
        # segment is disjoined
        self.assertEqual(0.0, rst[0, 3])
        # segment is contained
        self.assertEqual(10.0/30, rst[2, 1])
        # segment to left
        self.assertEqual(5.0/16, rst[2, 2])
        # segment to right
        self.assertEqual(6/15.0, rst[2, 3])

    def test_nms_detection(self):
        boxes = np.array([[10, 13],
                          [7, 11],
                          [5, 7],
                          [11, 12],
                          [9, 15]])
        scores = np.arange(boxes.shape[0])[::-1]
        # No score, NMS by iou
        idx_sol = [4, 3, 1, 2]
        bout, sout = segment.non_maxima_supression(boxes, None, 0.5)
        np.testing.assert_array_equal(bout, boxes[idx_sol, ...])
        # No score, NMS by overlap
        idx_sol = [4, 1, 2]
        bout, sout = segment.non_maxima_supression(boxes, None,
                                                   measure='overlap')
        np.testing.assert_array_equal(bout, boxes[idx_sol, ...])
        # With score, NMS by iou
        idx_sol = [0, 1, 2, 3]
        bout, sout = segment.non_maxima_supression(boxes, scores, 0.5)
        np.testing.assert_array_equal(bout, boxes[idx_sol, ...])
        # With score, NMS by overlap
        idx_sol = [0, 1, 2, 4]
        bout, sout = segment.non_maxima_supression(boxes, scores,
                                                   measure='overlap')
        np.testing.assert_array_equal(bout, boxes[idx_sol, ...])
