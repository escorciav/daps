#!/usr/bin/env python
"""

Generate action proposals for video

"""
import os
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import h5py
import numpy as np
import pandas as pd

from daps import C3D, DAPs
from daps.utils.segment import non_maxima_supression


def input_parser():
    description = ('Compute action proposals from its C3D feature '
                   'representation.')
    epilog = ('Note: It assumes that C3D features were densely extracted '
              'for all the frames.')
    p = ArgumentParser(description=description, epilog=epilog,
                       formatter_class=ArgumentDefaultsHelpFormatter)
    # Video arguments
    p.add_argument('-iv', '--video-name', required=True,
                   help='Name of video-id in your HDF5-file with C3D features')
    p.add_argument('-ic3d', '--c3d-hdf5', required=True,
                   help='HDF5 file with features for each video')
    p.add_argument('-imd', '--model-file', required=True,
                   help='npz file with sequence encoder parameters')
    p.add_argument('-iaf', '--anchors-hdf5', default='non-existent.hdf5',
                   help='HDF5 file with anchor segments')
    # Output arguments
    p.add_argument('-io', '--output-csv', default='',
                   help=('Filename to save proposals of video as CSV-file. If '
                         'empty "", it uses the same video-name'))
    p.add_argument('-c', '--clobber', action='store_true',
                   help='Overwrite outputs')
    # DAPs arguments
    p.add_argument('-ses', '--seq-encoder-stride', default=64, type=int,
                   help='Sliding stride for sequence encoder along the video')
    p.add_argument('-sel', '--seq-encoder-length', default=32, type=int,
                   help='Length of sequence encoder')
    p.add_argument('-sew', '--seq-encoder-width', default=256, type=int,
                   help='Number of hidden units per layer')
    p.add_argument('-sed', '--seq-encoder-depth', default=1, type=int,
                   help='Depth of sequence encoder')
    # Extra arguments
    p.add_argument('-vefr', '--c3d-f-res', default=16, type=int,
                   help='temporal resolution of C3D')
    p.add_argument('-vefs', '--c3d-f-stride', default=8, type=int,
                   help='temporal stride for C3D sampling')
    p.add_argument('-vept', '--c3d-pool-type', default='concat-32-mean',
                   help='Pooling strategy for C3D features')
    p.add_argument('-vefd', '--c3d-feat-dim', default=500, type=int,
                   help='Dimensionality of visual representation')
    p.add_argument('-vefi', '--c3d-feat-id', default='c3d_features',
                   help=('id used for HDF5-dataset corresponding to C3D '
                         'features'))
    return p


def main(video_name, c3d_hdf5, model_file, anchors_hdf5='non-existent',
         output_csv=None, clobber=False, seq_encoder_stride=64,
         num_proposals_per_seq_length=64, seq_encoder_length=32,
         seq_encoder_depth=1, seq_encoder_width=256, c3d_f_res=16,
         c3d_f_stride=8, c3d_pool_type='concat-32-mean', c3d_feat_dim=500,
         c3d_feat_id='c3d_features'):
    # Setup DAPs model
    # Infer receptive-field in terms of number of frames
    daps_receptive_field = seq_encoder_length * c3d_f_res

    # Visual Enconder
    print 'Setup interface with visual encoder'
    visual_encoder = C3D(c3d_hdf5, c3d_f_res, c3d_f_stride, c3d_pool_type,
                         c3d_feat_id)
    visual_encoder.open_instance()

    # Infer video length (it assumes C3D were densely extracted at every frame)
    num_c3d_features = visual_encoder.fobj[video_name][c3d_feat_id].shape[0]
    video_length = num_c3d_features + c3d_f_res
    # If num-frames less than DAPs-res, change t_stride
    if video_length < seq_encoder_length:
        raise ValueError('video-length < seq-encoder-time-steps.\nWe never '
                         'consider to create proposals for short clips')
    elif video_length < daps_receptive_field:
        warnings.warn(('video-length < DAPs-temporal-span. Increasing '
                       'sampling of c3d to compensate this.'), RuntimeWarning)
        visual_encoder.t_stride = int(num_c3d_features / seq_encoder_length)
        daps_receptive_field = video_length
        f_init_arr = np.arange(0, 1)
    else:
        f_init_arr = np.arange(0, video_length - daps_receptive_field + 1,
                               seq_encoder_stride)

    # Sequence Enconder
    # Load anchors file
    anchors = None
    if os.path.exists(anchors_hdf5):
        with h5py.File(anchors_hdf5) as f:
            anchors = f['anchors'].value

    print 'Setup sequence encoder'
    sequence_encoder = DAPs(num_proposals_per_seq_length, seq_encoder_length,
                            seq_encoder_depth, seq_encoder_width, c3d_feat_dim,
                            daps_receptive_field, anchors)
    print 'Loading sequence encoder model'
    sequence_encoder.load_model(model_file)
    print 'Compiling sequence encoder'
    sequence_encoder.compile()

    # Using DAPs
    print 'Reading C3D features'
    ve_representation = visual_encoder.read_feat_batch_from_video(
        video_name, f_init_arr, duration=daps_receptive_field)

    # Generate proposals along the whole video
    print 'Generating segments'
    proposals, score = sequence_encoder.retrieve_proposals(
        ve_representation, f_init_arr)

    # Post-processing
    print 'Post-processing segments'
    pp_proposals = proposals.reshape((-1, 2))
    pp_score = score.reshape(-1)
    nms_proposals, nms_score = non_maxima_supression(pp_proposals, pp_score)

    # Close visual encoder interface
    visual_encoder.close_instance()

    num_proposals = nms_proposals.shape[0]
    df_out = pd.DataFrame({'f-init': nms_proposals[:, 0],
                           'f-end': nms_proposals[:, 1],
                           'score': nms_score,
                           'video-name': [video_name] * num_proposals})
    # Dumping output
    if output_csv is not None:
        print 'Dumping results to disk'
        if len(output_csv) == 0:
            output_csv = video_name + '.csv'
        if not clobber and os.path.isfile(output_csv):
            raise ValueError('Existent output: {}'.format(output_csv))

        df_out.to_csv(output_csv, index=None, sep=' ')
    return df_out


if __name__ == '__main__':
    p = input_parser()
    main(**vars(p.parse_args()))
