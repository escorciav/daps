import h5py
import numpy as np

from daps.utils.pooling import concat1d


class C3D(object):
    """Simplify interaction with visual enconder (C3D network)

    Interface with an HDF5-file where you store the C3D features
    of your videos. Each video correspond to a HDF5-group which may have
    multiple features associated with it in the form of HDF5-dataset.
    This class just request 'c3d_features'.


    """
    def __init__(self, filename, f_res=16, f_stride=8,
                 pool_type='concat-32-mean', feat_id='c3d_features'):
        """Set the interface with your HDF5 file

        Parameters
        ----------
        filename : str.
            Full path to an hdf5 file.
        f_res : int, optional.
            Temporal receptive field of C3D encoder i.e. (resolution in terms
            of number of frames). Change it, if you use a different visual
            encoder.
        f_stride : int, optional.
            Temporal stride between features. We extracted C3D features densely
            for all the video frames. Therefore, we sample every 8 frames.
            Change it accordingly to your needs.
        pool_type : str, optional.
            Temporal pooling strategy over a bunch of features. You can choose
            among: None, '', 'mean', 'max', 'concat-2-mean/max'
        feat_id : str, optional.
            HDF5-dataset of interest for each video. Change it, if your
            HDF5-file does not support our definition.

        """
        self.filename = filename
        self.feat_id = feat_id
        self.fobj = None
        self.f_res = f_res
        self.f_stride = f_stride
        self.pool_type = pool_type

        with h5py.File(self.filename, 'r') as fobj:
            if not fobj:
                raise ValueError('Invalid type of file.')

    def open_instance(self):
        """Open file and keep it open till a close call.
        """
        self.fobj = h5py.File(self.filename, 'r')

    def close_instance(self):
        """Close existing h5py object instance.
        """
        if not self.fobj:
            raise ValueError('The object instance is not open.')
        self.fobj.close()
        self.fobj = None

    def read_feat(self, video_name, f_init=None, duration=None):
        """Stack C3D features in memory.

        Parameters
        ----------
        video-name : str.
            Video identifier.
        f_init : int, optional.
            Initial frame index. By default the feature is
            sliced from frame 1.
        duration : int, optional.
            duration in term of number of frames. By default
            it is set till the last feature.

        Returns
        -------
        pooled_feat : ndarray
            feature representation as 2-dim array [x, feat-dim]. The shape
            along the first dimension depends on the pooling strategy.
            For a pooling strategy equal to (None or ''), it yieds
            x = number-of-frames-of-interest.
            For a pooling strategy like 'mean' or 'max', it yieds x = 1.
            For a pooling strategy like 'concat-3-mean', it yieds x = 3.


        """
        if not self.fobj:
            raise ValueError('The object instance is not open.')

        f_end = None
        if f_init and duration:
            f_end = f_init + duration - self.f_size + 1
        elif (not f_init) and duration:
            f_end = duration - self.f_size + 1

        frames_of_interest = slice(f_init, f_end, self.f_stride)
        feat = self.fobj[video_name][self.feat_id][frames_of_interest, ...]
        pooled_feat = self._feature_pooling(feat)
        return pooled_feat

    def read_feat_batch_from_video(self, video_name, f_init_array,
                                   duration=512):
        """Read batch of C3D features from a video.

        Parameters
        ----------
        video-name : str.
            Video identifier.
        f_init_array : list or 1d-ndarray
            list of initial frames.
        duration : int.
            Segment size.

        Returns
        -------
        feat_stack : ndarray
            stack feature representation as 3dim array of shape
            [len(f_init_array), x, feat-dim]. Check feat_stack for details
            about value of x.

        """
        if not self.fobj:
            raise ValueError('The object instance is not open.')
        if isinstance(f_init_array, np.ndarray) and f_init_array.ndim > 1:
            raise ValueError('Use a 1dim ndarray ofr f_init_array')
        # Sanitize.
        f_init_array = np.array(f_init_array).astype(int)
        duration = int(duration)

        # Load all features associated to video-name.
        raw_feat_stack = self.fobj[video_name][self.feat_id].value
        n_segments = len(f_init_array)

        # Set feat stack size.
        d = raw_feat_stack.shape[1]
        if self.pool_type is None or self.pool_type == '':
            m = (duration - self.f_res)/self.f_stride + 1
        elif self.pool_type == 'mean' or self.pool_type == 'max':
            m = 1
        elif 'concat' in self.pool_type:
            _, levels, pool_type = self.pool_type.split('-')
            m = int(levels)
        else:
            raise ValueError('Incorrect pool_type')
        feat_stack = np.empty((n_segments, m, d))

        # Iterate over each segment.
        for i, f_init in enumerate(f_init_array):
            frames_of_interest = slice(
                f_init, f_init + duration - self.f_res + 1, self.f_stride)
            feat_stack[i, ...] = self._feature_pooling(
                raw_feat_stack[frames_of_interest, :])

        return feat_stack

    def _feature_pooling(self, x):
        """Compute pooling of a feature vector.

        Parameters
        ----------
        x : ndarray.
            [m, d] array of features.m is the number of features and
            d is the dimensionality of the feature space.

        Returns
        -------
        pooled_x : ndarray
            [n, d] 2-dim ndarray of feature vector representation after
            applying pooling over first dimension.

        Notes
        -----
        1. There is no guarantee that output is a contiguous arrya.

        """
        if x.ndim != 2:
            raise ValueError('Invalid input ndarray. Input must be [mxd].')
        m, d = x.shape

        if self.pool_type == '' or self.pool_type is None:
            return x
        elif self.pool_type == 'mean':
            return x.mean(axis=0, keepdims=True)
        elif self.pool_type == 'max':
            return x.max(axis=0, keepdims=True)
        elif 'concat' in self.pool_type:
            _, level, pool_type = self.pool_type.split('-')
            x = concat1d(x, int(level), pool_type)
            return x.reshape((-1, d))
