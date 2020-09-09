#!/usr/bin/env python

# Compare the distribution of corners, actual and reconstructed

import os
import sys

import tensorflow as tf
import numpy
import itertools

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.append("%s/../" % os.path.dirname(__file__))
from cornerModel import cornerModel

sys.path.append("%s/../../dataset" % os.path.dirname(__file__))
from makeDataset import getImageDataset
from makeDataset import getCornersDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=25)
# Set nimages to a small number for fast testing
parser.add_argument(
    "--nimages",
    help="No of test cases to look at",
    type=int,
    required=False,
    default=None,
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

# Get the corners - original and recovered
testImages = getImageDataset(purpose="test", nImages=args.nimages)
testNumbers = getCornersDataset(purpose="test", nImages=args.nimages)
testData = tf.data.Dataset.zip((testImages, testNumbers))
original = []
recovered = []
for testCase in testData:
    image = testCase[0]
    orig = testCase[1]
    encoded = seeker(tf.reshape(image, [1, 1024, 768, 3]), training=False)
    for tidx in range(orig.shape[0]):
        original.append(orig[0:])
        recovered.append(encoded[0, :])

# Plot original corner points on the left, recovered on the right
#  on the right
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
ax_full.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="grey")
)

# Original
ax_original = fig.add_axes([0.02, 0.015, 0.47, 0.97])
ax_original.set_axis_off()
ax_original.set_xlim([0, 1])
ax_original.set_ylim([0, 1])
ax_original.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="white")
)
for cset in original:
    fc = ("red", "blue", "orange", "black")
    for point in range(4):
        ax_original.add_patch(
            matplotlib.patches.Circle(
                (cset[point * 2], cset[point * 2 + 1]),
                radius=0.005,
                facecolor=fc[point],
                edgecolor=fc[point],
                linewidth=0.1,
                alpha=0.05,
            )
        )


# Plot encoded using same method as original plot
ax_encoded = fig.add_axes([0.51, 0.015, 0.47, 0.97])
ax_encoded.set_xlim([0, 1])
ax_encoded.set_ylim([0, 1])
ax_encoded.set_axis_off()
ax_encoded.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="white")
)
for cset in recovered:
    fc = ("red", "blue", "orange", "black")
    # print(cset)
    for point in range(4):
        ax_encoded.add_patch(
            matplotlib.patches.Circle(
                (cset[point * 2], cset[point * 2 + 1]),
                radius=0.005,
                facecolor=fc[point],
                edgecolor=fc[point],
                linewidth=0.1,
                alpha=0.05,
            )
        )


# Render the figure as a png
fig.savefig("compare.png")
