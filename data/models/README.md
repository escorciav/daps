This folder contains DAPs models.

# T512K64 THUMOS14

Model used in the camera ready of our ECCV-work. It was trained on 80% of THUMOS14 validation set with temporal annotations.

- number of anchors: 64

- temporal receptive field: 512

*Details*

format: This file was save [numpy.savez](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) using the version reported in our environment YAML file.

order: the order of the arrays comes from the function [get_all_param_values](http://lasagne.readthedocs.io/en/latest/modules/layers/helper.html#lasagne.layers.get_all_param_values) applied on our [sequence encoder](https://github.com/escorciav/daps/blob/master/daps/sequence_encoder.py#L11) with default values.

## T512K64 anchors THUMOS14

anchors used to train the model [T512K64 THUMOS14](#T512K64-THUMOS14).

*Details*

format: This file is an HDF5 file with a unique dataset called *anchors*.

# T256K16 THUMOS14

Model used in an early-stage version of our work. It was trained on 80% of THUMOS14 validation set with temporal annotations.

- number of anchors: 16

- temporal receptive field: 256

*Details*

format: This file was save [numpy.savez](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) using the version reported in our environment YAML file.

## T256K16 anchors THUMOS14

anchors used to train the model [T256K16 THUMOS14](#T246K16-THUMOS14).

format: This file is an HDF5 file with a unique dataset called *anchors*.

# pca_c3d_fc7_thumos14.hdf5

Results of PCA analysis of C3D (FC7) representation of videos in THUMOS14.

*Details*

format: This file is an HDF5 file with three datasets *S*, *U*, *x_mean*.

*Usage*

Given a video-clip with a visual representation **x**, you can reduce its dimensionality up to 500 dimension by doing:

```python
import numpy as np

# x = define-your-feature-vector-here
num_red_dim = 500
x_red = np.dot(x - x_mean, U[:, :num_red_dim])
```

> Note for curious user: you can easily plug this operation as a lasagne layer :wink:
