# Samples

This folder contains a sample of the HDF5-file required by our program.

We also provide an example to save this file [here](#Example-to-write-HDF5)

## List of items

### c3d_after_pca.hdf5

HDF5-file with C3D representation (from fc7) of two videos.
Each videos is associated with an [HDF5-group](http://docs.h5py.org/en/latest/high/group.html) and its visual representation is stored as an [HDF5-dataset](http://docs.h5py.org/en/latest/high/dataset.html) called *c3d_features*.

Below, you can find the tree structure illustration of this file:

```
|-- c3d_after_pca.hdf5
|   |-- video_test_0000526 (Group)
|   |   |-- c3d_features (Dataset)
|   |-- video_test_0000541 (Group)
|   |   |-- c3d_features (Dataset)
```

## Example to write HDF5

You can saved your HDF5 file using something similar to this:

```python
"""Edit the following three lines
list_of_video_ids = [fill-this-variable]
list_of_c3d_data = [fill-this-variable]
new_file = 'fill-name-of-your-file.hdf5'
"""Remove this and the first line ;)

with h5py.File(new_file, 'w') as fid:
    for i, video_id enumerate(list_of_video_ids):
        grp = fid.create_group(video_id)
        data = list_of_c3d_data[i]
        grp.create_dataset('c3d_features', data=data.astype(np.float32),
                           chunks=True, compression="gzip", compression_opts=9)
```

>Note: The previous snippet won't run if you do not edit the first three lines and remove the lines starting with `"""`.

For an example of how to read this file, take a look of our [visual enconder interface](https://github.com/escorciav/daps/blob/master/daps/visual_encoder.py).
It has plenty of comments but it is not as handy as the previous snippet.
