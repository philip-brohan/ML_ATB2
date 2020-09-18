#!/usr/bin/env python

# Take one of the test images and deskew it - map the data grid to a standard size,
#   position and orientation.
# Save the deskewed image as a tensor for further work.

import os
import sys

import tensorflow as tf
import numpy
import itertools

from scipy.interpolate import RectBivariateSpline

sys.path.append("%s" % os.path.dirname(__file__))
from cornerModel import cornerModel

sys.path.append("%s/../dataset" % os.path.dirname(__file__))
from makeDataset import getImageDataset
from makeDataset import getCornersDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=50)
parser.add_argument(
    "--image", help="Test image number", type=int, required=False, default=0
)
args = parser.parse_args()

# Set up the model and load the weights at the chosen epoch
seeker = cornerModel()
weights_dir = ("%s/ML_ATB2/models/find_corners/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch - 1,
)
load_status = seeker.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# Get image number args.image - need both training and test set
testImage = getImageDataset(purpose="sample",selection=args.image)
testImage = testImage.batch(1)
originalImage = next(itertools.islice(testImage, 0, 1))
testNumbers = getCornersDataset(purpose="sample",selection=args.image)
testNumbers = testNumbers.batch(1)
original = next(itertools.islice(testNumbers, 0, 1))
original = original.numpy()

# Run that test image through the transcriber
encoded = seeker.predict_on_batch(originalImage)

# Map the skewed original to a standard rectangle

# Map an x,y (0-1) location in the standard rectangle, to
#  a location in the original image
def locMap(x, y, encoded):
    # Point fraction y of the way up the left side
    #  of the rotated grid
    fryd = (y - 0.05) / 0.9  # Allow for 0.05 padding
    lptX = encoded[0] * fryd + encoded[4] * (1 - fryd)
    lptY = (1 - encoded[1]) * fryd + (1 - encoded[5]) * (1 - fryd)
    # Same for the right side
    rptX = encoded[2] * fryd + encoded[6] * (1 - fryd)
    rptY = (1 - encoded[3]) * fryd + (1 - encoded[7]) * (1 - fryd)
    # Point fraction x of the way between the two side points
    frxd = (x - 0.05) / 0.9  # Allow for 0.05 padding
    sptX = lptX * frxd + rptX * (1 - frxd)
    sptY = lptY * frxd + rptY * (1 - frxd)
    return (sptX, sptY)


# Interpolator for the skewed image
x = numpy.arange(0, 1, 1 / 768)
y = numpy.arange(0, 1, 1 / 1024)
z = tf.reshape(originalImage, [1024, 768, 3])[:, :, 0]
interpolator = RectBivariateSpline(
    y, x, z, bbox=[None, None, None, None], kx=3, ky=3, s=0
)

# Standard rectangle size is 768*512
standardised = numpy.zeros([512, 768, 3])
for x in range(768):
    for y in range(512):
        (x2, y2) = locMap(x / 768, y / 512, encoded[0, :])
        st = interpolator(y2, x2)[0][0]
        for z in range(3):
            standardised[511 - y, 767 - x, z] = st

# Output the tensor
opdir = "%s/ML_ATB2/tensors/standardised/" % os.getenv("SCRATCH")
if not os.path.isdir(opdir):
    try:  # These calls sometimes collide
        os.makedirs(opdir)
    except FileExistsError:
        pass

ict = tf.convert_to_tensor(standardised, numpy.float32)
# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file("%s/%04d.tfd" % (opdir, args.image), sict)
