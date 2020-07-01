#!/usr/bin/env python

# Convert all the fake rainfall image numbers to tensors for ML model training

# This script does not run the commands - it makes a list of commands
#  (in the file 'run_n2t.txt') which can be run in parallel.

import os

rootd = "%s/OCR-fake/numbers/" % os.getenv("SCRATCH")


f = open("run_n2t.sh", "w+")

for doci in range(10000):
    if os.path.isfile(
        "%s/ML_ATB2/tensors/numbers/%04d.tfd" % (os.getenv("SCRATCH"), doci)
    ):
        continue
    cmd = ('./numbers_to_tensor.py --docn="%04d"\n') % doci
    f.write(cmd)

f.close()
