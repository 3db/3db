# Docker [Experimental]

Here we provide an (experimental) docker instance for using 3DB.

## Usage

1. Clone this repo: ``git clone https://github.com/3db/3db.git /path/to/3db``
2. Build the docker image: ``docker build -t /path/to/3db/docker``
3. Run the instance: ``docker run -it -v /path/to/3db/:/3db threedb``
4. In the instance, run: ``conda activate threedb`` to activate the env
5. You're all set! You are now in a docker instance with blender and threedb
   installed and can run [any](TODO) [of the](TODO) [demos](TODO) using the
   instructions therein.
