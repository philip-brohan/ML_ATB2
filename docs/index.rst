Machine Learning for OCR of weather records
===========================================

Datasets of historical weather observations are vital to our understanding of climate change and variability, and improving those datasets means transcribing millions of observations - converting paper records into a digital form. Doing such transcription manually is `expensive and slow <http://brohan.org/transcription_methods_review/>`_, and we have a backlog of millions of pages of potentially valuable records which have never been transcribed. We would dearly like a cheap, fast, software tool for extracting weather observations from (photographs of) archived paper documents. No such system currently exists, but recent developments in machine learning methods and image analysis tools suggest that it might now be possible to create one.

This is an attempt to create such a tool: specifically it is an attempt to use the `TensorFlow <https://www.tensorflow.org/>`_ machine learning toolkit to solve an `idealised document auto-transcription benchmark <http://brohan.org/Auto-transcription-benchmark-2-Fake-data/index.html>`_.

.. toctree::
   :maxdepth: 1

   Getting started <preface>
   Preparing the input data <prepare_data>

Then we can experiment with different model designs until we find one that is successful in transcription:

.. toctree::
   :maxdepth: 1

   1) A deep convolutional transcriber <models/deep_convolutional_transcriber/index>

   
.. toctree::
   :maxdepth: 1

   Authors and acknowledgements <credits>

This dataset is distributed under the terms of the `Open Government Licence <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/>`_. Source code included is distributed under the terms of the `BSD licence <https://opensource.org/licenses/BSD-2-Clause>`_.

