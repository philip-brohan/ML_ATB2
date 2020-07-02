Getting started
===============

This project is kept under version control in a `git repository <https://en.wikipedia.org/wiki/Git>`_. The repository is hosted on `GitHub <https://github.com/>`_ (and the documentation made with `GitHub Pages <https://pages.github.com/>`_). The repository is `<https://github.com/philip-brohan/ML_ATB2>`_.

If you are familiar with GitHub, you already know what to do (fork or clone `the repository <https://github.com/philip-brohan/ML_ATB2>`_): If you'd prefer not to bother with that, you can download the whole thing as `a zip file <https://github.com/philip-brohan/ML_ATB2/archive/master.zip>`_.

The software in this repository operate on data provided by `another repository <http://brohan.org/Auto-transcription-benchmark-2-Fake-data/index.html>`_. You'll need that too, it has its own installation instructions.

As well as downloading the software, some setup is necessary to run them successfully:

These scripts need to know where to put their output files. They rely on an environment variable ``SCRATCH`` - set this variable to a directory with plenty of free disc space.

These scripts will only work in a `python <https://www.python.org/>`_ environment with the appropriate python version and libraries available. I use `conda <https://docs.conda.io/en/latest/>`_ to manage the required python environment - which is specified in a yaml file:

.. literalinclude:: ../environments/ml_atb2_gpu.yml

Install `anaconda or miniconda <https://docs.conda.io/en/latest/>`_, `create and activate the environment in that yaml file <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs>`_, and all the scripts in this repository should run successfully.

The environment shown above installs the GPU-enabled version of the `Tensorflow libraries <https://www.tensorflow.org/>`_, and requires an (NVIDIA) GPU to run. If you have no GPU, replace ``tensorflow-gpu`` in the specification with ``tensorflow`` and it will work just as well on a CPU-only system (model training will be much slower). It can be convenient to run the data preparation and model validation steps on CPU-only systems, but for the model training steps a GPU is very desirable. 


