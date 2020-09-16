#!/usr/bin/env python

# Take one of the test images and deskew it - map the data grid to a standard size,
#   position and orientation.

import os
import sys

import tensorflow as tf
import numpy
import itertools

from scipy.interpolate import RectBivariateSpline

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.append("%s/../.." % os.path.dirname(__file__))
from cornerModel import cornerModel

sys.path.append("%s/../../../dataset" % os.path.dirname(__file__))
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

# Get test case number args.image
testImage = getImageDataset(purpose="test", nImages=args.image + 1)
testImage = testImage.batch(1)
originalImage = next(itertools.islice(testImage, args.image, args.image + 1))
testNumbers = getCornersDataset(purpose="test", nImages=args.image + 1)
testNumbers = testNumbers.batch(1)
original = next(itertools.islice(testNumbers, args.image, args.image + 1))
original = original.numpy()

# Run that test image through the transcriber
encoded = seeker.predict_on_batch(originalImage)

# Plot original image on the left, deskewed image on the right.
fig = Figure(
    figsize=(16.34, 10.56),
    dpi=100,
    facecolor="white",
    edgecolor="black",
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
# Paint the background white - why is this needed?
ax_full = fig.add_axes([0, 0, 1, 1])
ax_full.set_xlim([0, 1])
ax_full.set_ylim([0, 1])
ax_full.set_aspect("auto")
ax_full.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="grey")
)
fc = ("red", "blue", "orange", "black")

# Original
ax_original = fig.add_axes([0.02, 0.015, 0.47, 0.97])
ax_original.set_axis_off()
ax_original.set_xlim([0, 768])
ax_original.set_ylim([1024, 0])
ax_original.set_aspect("auto")
ax_original.imshow(
    tf.reshape(originalImage, [1024, 768, 3]),
    aspect="auto",
    origin="upper",
    interpolation="nearest",
)
cset = encoded[0, :]
for point in range(4):
    ax_original.add_patch(
        matplotlib.patches.Circle(
            (cset[point * 2] * 768, (1 - cset[point * 2 + 1]) * 1024),
            radius=10,
            facecolor=fc[point],
            edgecolor=fc[point],
            linewidth=0.1,
            alpha=0.8,
        )
    )

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


def lM2(x, y, encoded):
    return (x, y / 2 + 0.2)


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

# Plot encoded using same method as original plot
ax_encoded = fig.add_axes([0.51, 0.315, 0.47, 0.485])
ax_encoded.set_axis_off()
ax_encoded.set_xlim([0, 768])
ax_encoded.set_ylim([512, 0])
ax_encoded.set_aspect("auto")
ax_encoded.imshow(
    standardised, aspect="auto", origin="upper", interpolation="nearest",
)

# Render the figure as a png
fig.savefig("standardised.png")
