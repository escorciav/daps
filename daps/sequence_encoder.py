import lasagne
import numpy as np
import theano
from lasagne.layers import DenseLayer, InputLayer, LSTMLayer, SliceLayer
from lasagne.layers import get_output
from lasagne.utils import floatX

from daps.utils.segment import format as segment_format


class DAPs(object):
    """Deep Action Proposal (seq. enconder & proposal generation)
    """
    def __init__(self, num_outputs=64, seq_length=32, depth=1,
                 width=256, input_size=500, receptive_field=512):
        """Initialize DAPs architecture

        Parameters
        ----------
        num_outputs : int, optional
            Number of proposals per sequence length
        seq_length : int, optional
            Number of time-steps
        depth : int, optional
            Number of LSTM layers (stack on top of each other)
        width : int, optional
            Number of hidden units of each LSTM layer
        input_size : int, optional
            Dimension of feature vector
        receptive_field : int, optional
            Receptive field in terms of number of frames

        """
        self.num_outputs = num_outputs
        self.seq_length = seq_length
        self.depth = depth
        self.width = width
        self.input_size = input_size
        self.model = None
        self.receptive_field = receptive_field
        self._build()

    def _build(self):
        """Build architecture
        """
        self.network = []
        self.network.append(InputLayer(shape=(None, self.seq_length,
                                              self.input_size),
                                       name='input'))
        self.input_var = (self.network[0]).input_var

        # Hidden layers
        tanh = lasagne.nonlinearities.tanh
        for _ in range(self.depth):
            self.network.append(
                LSTMLayer(self.network[-1], self.width, nonlinearity=tanh))

        # Retain last-output state
        self.network.append(SliceLayer(self.network[-1], -1, 1))

        # Output layer
        sigmoid = lasagne.nonlinearities.sigmoid
        self.loc_var = DenseLayer(self.network[-1], self.num_outputs * 2)
        self.conf_var = DenseLayer(self.network[-1], self.num_outputs,
                                   nonlinearity=sigmoid)

    def compile(self, **kwargs):
        """Compile theano function

        Parameters
        ----------
        kwargs : dict
            Optional theano configuration

        """
        if callable(self.model):
            print 'Model is already compile'
            return None
        loc_expr, conf_expr = get_output((self.loc_var, self.conf_var),
                                         deterministic=True)
        self.model = theano.function([self.input_var],
                                     [loc_expr, conf_expr])

    def forward_pass(self, input_data):
        """Generate proposals for bunch of clips

        Parameters
        ----------
        input_data : ndarray
            3d-ndarray of size [n_streams, seq-length, input-dim]

        Returns
        -------
        loc : ndarray
            2d-ndarray of size [n_streams, 2 * self.num_outputs]
        conf : ndarray
            2d-ndarray of size [n_streams, self.num_outputs]

        Raises
        ------
        ValueError
            - Model has not been compiled
            - input_data is not a 3d-ndarray
            - input_data.shape[1] is different than seq-length
            - input_data.shape[2] is different than input-size

        """
        if not callable(self.model):
            raise ValueError('Compile model before feeding data up')
        if input_data.ndim != 3:
            raise ValueError('Input data must be 3-dim array')
        if input_data.shape[1] != self.seq_length:
            raise ValueError('Incorrect input data size along 2nd dimension.')
        if input_data.shape[2] != self.input_size:
            raise ValueError('Incorrect input data size along 3rd dimension.')

        loc, conf = self.model(input_data)
        return loc, conf

    def load_model(self, filename):
        """Set parameters of DAPs model

        Parameters
        ----------
        filename : str
            Fullpath of npz-file with weights of the network

        """
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values([self.loc_var, self.conf_var],
                                                param_values)

    def retrieve_proposals(self, c3d_stack, f_init_array):
        """Retrieve proposals for multiple streams.

        Parameters
        ----------
        c3d_stack : ndarray
            3d-ndarray [num-streams, seq-length, input-size] with visual
            encoder representation of each stream.
            Note that the first dimension is sequence agnostic so you can
            push as many videos as your HW allows it.
        f_init_array : ndarray.
            1d-ndarray with initial frame of each stream.

        Returns
        -------
        proposals : ndarray
            3d-ndarray [num-streams, num-outputs, 2] with proposal locations in
            terms of f-init, f-end.
        conf : ndarray
            2d-ndarray [num-streams, num-outputs] action likelihood of each
            proposal

        Raises
        ------
        ValueError
            Mistmatch between c3d_stack.shape[0] and f_init_array.size

        """
        if c3d_stack.ndim == 2 and c3d_stack.shape[0] == self.seq_length:
            c3d_stack = c3d_stack[np.newaxis, ...]
        if c3d_stack.shape[0] != f_init_array.size:
            raise ValueError('Mismatch between c3d_stack and f_init_array')
        n_streams = c3d_stack.shape[0]

        loc, score = self.forward_pass(floatX(c3d_stack))

        # Clip proposals inside receptive field
        loc.clip(0, 1, out=loc)
        loc *= self.receptive_field

        # Shift center to absolute location in the video
        loc = loc.reshape((n_streams, -1, 2))
        loc[:, :, 0] += f_init_array.reshape((n_streams, 1))

        # Transform center 2 boundaries
        proposals = np.reshape(
            segment_format(loc.reshape((-1, 2)), 'c2b').astype(int),
            (n_streams, -1, 2))
        return proposals, score
