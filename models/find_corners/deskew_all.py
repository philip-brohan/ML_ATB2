#!/usr/bin/env python

# Standardise all the fake rainfall images using fitted corners.

# This script does not run the commands - it makes a list of commands
#  (in the file 'run_std.txt') which can be run in parallel.

import os

f = open("run_std.sh", "w+")

for doci in range(10000):
    if os.path.isfile(
        "%s/ML_ATB2/tensors/standardised/%04d.tfd" % (os.getenv("SCRATCH"), doci)
    ):
        continue
    cmd = "./deskew_image.py --image=%d --epoch=150\n" % doci
    f.write(cmd)

f.close()
