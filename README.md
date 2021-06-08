# 3DB: A framework for analyzing computer vision models with simulated data

[Paper](https://arxiv.org/abs/2106.03805)

[Quickstart guide](https://3db.github.io/3db/usage/quickstart.html)

[Blog post](http://gradientscience.org/3db)

## Installation

Follow instructions on: https://github.com/3db/installers

## Complete demo

For detailed, step by step demonstration of the usage of the framework, please visit: https://github.com/3db/demo

## Documentation

You can find in-depth documentation for the package [here](https://3db.github.io/3db), including a
quickstart guide, an explanation on the internal layout of the package, as well
as guides for extending 3DB to new modalities, tasks, and output formats.

## Primer data

We offer data for users to quickly get started 3DB and reproduce the results of the paper mentioned below. It is available on dropbox at this link: https://www.dropbox.com/s/2gdprhp8jvku4zf/threedb_starting_kit.tar.gz?dl=0. (One can use `wget` to download it).

It contains:

- 3D models
- Environments: our studio and HDRI backgrounds
- Replacement textures to use with `threedb.controls.blender.material`
- Licensing/credits for the data

## Citation

If you find 3DB helpful, please cite it as:
```
@inproceedings{leclerc2021three,
    title={3DB: A Framework for Debugging Computer Vision Models},
    author={Guillaume Leclerc, Hadi Salman, Andrew Ilyas, Sai Vemprala, Logan Engstrom, Vibhav Vineet, Kai Xiao, Pengchuan Zhang, Shibani Santurkar, Greg Yang, Ashish Kapoor, Aleksander Madry},
    year={2021},
    booktitle={Arxiv preprint arXiv:2106.03805}
}
```
