#!/usr/bin/env python

# Convert all the fake rainfall images to tensors for ML model training

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os

rootd = "%s/OCR-fake/images/" % os.getenv("SCRATCH")


f = open("run_i2t.sh", "w+")

for doci in range(10000):
    if os.path.isfile(
        "%s/ML_ATB2/tensors/images/%04d.tfd" % (os.getenv("SCRATCH"), doci)
    ):
        continue
    cmd = ('./image_to_tensor.py --docn="%04d"\n') % doci
    f.write(cmd)

f.close()
