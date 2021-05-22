from setuptools import setup
import distutils.cmd
import os
import distutils.log
import setuptools
import setuptools.command.build_py
import subprocess
from os import path
from subprocess import check_output

DASHBOARD_REPO = "https://github.com/3db/dashboard.git"

folder = path.dirname(path.abspath(__file__))
with open(path.join(folder, 'requirements.txt'), 'r') as handle:
    requirements = handle.readlines()


class BuildDashboardCommand(distutils.cmd.Command):

  description = 'build the 3db dashboard'
  user_options = [
      # The format is (long option, short option, description).
      ('dashboard-repo=', None, 'repo to get the dasboard from'),
  ]

  def initialize_options(self):
    """Set default values for options."""
    # Each user option must be listed here with their default value.
    self.dashboard_repo = DASHBOARD_REPO

  def finalize_options(self):
    """Post-process options."""
    pass


  def run(self):
    """Run command."""
    command = ['/bin/bash', './build_dashboard.sh', 'self.dashboard_repo}']
    command = ['/bin/bash', path.join(os.getcwd(), 'build_dashboard.sh'), self.dashboard_repo]
    self.announce(
        'Running command: %s' % str(command),
        level=distutils.log.INFO)
    r = subprocess.check_output(command)
    print("r",r)


setup(name='threedb-preview',
      version='0.0.2',
      description='A framework for analyzing computer vision models with simulated data ',
      url='https://github.com/3db/3db',
      author='3DB authors',
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
        'flask',
        'flask_cors',
        'tensorboard',
        'mathutils',
        'typeguard',
        'pyyaml',
        'flask-compress'],
      packages=['threedb'],
      cmdclass={
          'build_dashboard': BuildDashboardCommand
      },
      include_package_data=True,
      zip_safe=False)