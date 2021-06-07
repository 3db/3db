from setuptools import setup, find_packages
import distutils.cmd
import os
import distutils.log
import setuptools
import setuptools.command.build_py
import subprocess
from os import path
from subprocess import check_output

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='threedb-preview',
      version='0.0.7',
      description='A framework for analyzing computer vision models with simulated data ',
      url='https://github.com/3db/3db',
      author='3DB authors',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author_email='leclerc@mit.edu',
      license='MIT',
      install_requires=[
        'torch>=1.7.0',
        'torchvision',
        'cox',
        'robustness',
        'kornia',
        'scikit-image',
        'orjson',
        'opencv-python',
        'zmq',
        'tensorboard',
        'ipdb',
        'flatten-dict',
        'tqdm',
        'pytest',
        'pytest-xdist',
        'tensorboard',
        'typeguard',
        'pyyaml'],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
