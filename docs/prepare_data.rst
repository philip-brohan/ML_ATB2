Process benchmark data into tensors for model building
======================================================

The data to be modelled are taken from `Auto-transcription benchmark 2: Fake data <http://brohan.org/Auto-transcription-benchmark-2-Fake-data/index.html>`_ and consist of 10,000 png images and 10,000 python pickle files - each pickle file containing an array of 360 digits (0-9). To model these with `TensorFlow <https://www.tensorflow.org/>`_ we need to convert both the images and the pickled arrays into `tensors <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_ and it's much more efficient to do this in a preprocessing step than in-line during model training.

So we need a script to convert a png image file into a tensor file, a script to convert a pickled array into a tensor, and a script to run each of these 10,000 times - once for each case in the benchmark dataset.

.. toctree::
   :maxdepth: 1

   Script to convert an image <data/image_to_tensor>
   Script to convert a pickled array <data/numbers_to_tensor>
   Script to run these 10,000 times <data/convert_all>

The resulting serialised tensor files will take about 100 Gb of disc space (tensors are a space-inefficient way to store images).

Then to present all those tensors to a ML model for training we need to package them as a `TensorFlow Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_:

.. toctree::
   :maxdepth: 1

   Function providing benchmark data as a tf.data.Dataset <data/files_to_dataset>
