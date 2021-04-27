# 3DB: A framework for analyzing computer vision models with simulated data

## Installation

Follow instructions on: https://github.com/3db/installers

## Complete demo

For detailed, step by step demonstration of the usage of the framework, please visit: https://github.com/3db/demo

## Documentation

You can find in-depth documentation for the package [here](LINK), including a
quickstart guide, an explanation on the internal layout of the package, as well
as guides for extending 3DB to new modalities, tasks, and output formats.

## Citation

If you find 3DB helpful, please cite it as:
```
@inproceedings{leclerc2021three,
    title={3DB: A Framework for Chungus Chungus},
    author={Guillaume Leclerc, Hadi Salman, Andrew Ilyas, Sai Vemprala, ..., Ashish Kapoor, Aleksander Madry},
    year={2021},
    booktitle={Arxiv preprint arXiv:TODO}
}
```


3 Tmux sessions running the client, server, and tensorboard will be started for you!

## Basic Data Collection
This shows you how to collect a dataset using blender to analyze the quality of the models generated. We assume that the blender environments are in `$BLENDER_ENVS`, models are in `$BLENDER_MODELS`, and the desired output directory is in `$OUTDIR`.

- Example enviroment is [here](environments/).
- 3DModels can be downloaded from azure storage:
    - Get `azcopy` which will be usefull to download the auto-generated dataset later on too.
        ```
        wget -O azcopy.tar https://aka.ms/downloadazcopy-v10-linux \
        tar -xzvf azcopy.tar && cp azcopy_linux_amd64_10.6.0/azcopy ~/ && rm -rf azcopy_linux_amd64_10.6.0 azcopy.tar
        ```
    - Run the following command to download the models:
        ```
        azcopy cp 'https://objectsim.blob.core.windows.net/sandbox/blender_models?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2021-09-04T17:58:15Z&st=2020-09-04T09:58:15Z&spr=https,http&sig=CPuCBI%2FtMrSlzytVt7UioVkyKC9%2Fetp5XqTC2rtjino%3D' '/models' --recursive
        ```
