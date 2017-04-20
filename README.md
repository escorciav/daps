# Deep Action Proposals for Videos

Temporal Action Proposals for long untrimmed videos.

DAPs architecture allows to retrieve segments from long videos where it is likely to find actions with high recall very quickly.

![pull arch figure][image-modal]

## Welcome

Welcome to our repo! This project hosts a simple, handy interface to generate segments where it is likely to find actions in your videos.

If you find any piece of code valuable for your research please cite this work:

```
@Inbook{Escorcia2016,
author="Escorcia, Victor and Caba Heilbron, Fabian and Niebles, Juan Carlos and Ghanem, Bernard",
editor="Leibe, Bastian and Matas, Jiri and Sebe, Nicu and Welling, Max",
title="DAPs: Deep Action Proposals for Action Understanding",
bookTitle="Computer Vision -- ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III",
year="2016",
publisher="Springer International Publishing",
address="Cham",
pages="768--784",
isbn="978-3-319-46487-9",
doi="10.1007/978-3-319-46487-9_47",
url="http://dx.doi.org/10.1007/978-3-319-46487-9_47"
}
```

If you like this project, give us a :star: in the github banner :wink:.

## Installation

1. Ensure that you have [gcc](https://gcc.gnu.org/), [conda](ihttp://conda.pydata.org/docs/index.html), [CUDA and CUDNN](https://developer.nvidia.com/cuda-downloads) (optional).

2. Clone our repo, `git clone https://github.com/escorciav/daps/`.

3. Go to our project folder and type `bash install.sh`.

*Notes*

- Our implementation uses Theano. It is tested with gcc but as long as Theano supports your desired compiler, go ahead.

- In case you don't want to use conda, our python dependencies are [here](https://github.com/escorciav/daps/blob/master/requirements.txt). A complete list of dependecies is [here](https://github.com/escorciav/daps/blob/master/environment_x64.yml).

- Do you like [environment-modules](http://modules.sourceforge.net/)? we provide bash scripts to [activate](https://github.com/escorciav/daps/blob/master/activate.sh) or [deactivate](https://github.com/escorciav/daps/blob/master/deactivate.sh) the environment. Personalize them :wink:.

## What can you find?

- [Pre-trained models](https://github.com/escorciav/daps/tree/master/data/models). Our generalization experiment suggests that you may expect decent results for other kind of action classes with similar lengths. Check out the models trained on the validation set of THUMOS14.

- [Pre-computed action proposals](https://github.com/escorciav/daps/wiki/FAQs#can-you-share-the-daps-proposal-results). Take a look at our results if you are interested in comparisons or building cool algorithms on top of our outputs.

- [Code for retrieving proposals in new videos](https://github.com/escorciav/daps/blob/master/tools/generate_proposals.py). Check out our program to retrieve proposals from your video.

## Do you want to try?

1. Download the C3D representation of a couple of videos from [here](https://github.com/escorciav/daps/blob/master/data/samples/c3d_after_pca.hdf5).

2. Download our [model](https://github.com/escorciav/daps/blob/master/data/models/T512K64_thumos14.npz).

3. Go to our project folder

3. Activate our conda environment. We reduce your choices to one of the following:

  - Execute `source activate daps-eccv16`, for conda users.

  - Execute `./activate.sh`, for conda and environment-modules users.

  - Ensure that our package is in your `PYTHONPATH`, python users.

  > Note for **new** environment-modules users: You must personalize the script to [activate](https://github.com/escorciav/daps/blob/master/activate.sh) the environment, otherwise it will fail.

4. Execute: `tools/generate_proposals.py -iv video_test_0000541 -ic3d [path-to-c3d-of-videos] -imd [path-our-model]`

## Questions

Please visit our [FAQs](https://github.com/escorciav/daps/wiki/FAQs), if you have any doubt. In case that your question is not there, send us an email.

<!--Images-->
[image-modal]: https://escorciav.github.io/img/portfolio/kaust16_modal.png
