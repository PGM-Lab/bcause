#!/usr/bin/python3


import subprocess
import datetime
import os
import sys


from datetime import datetime
from pathlib import Path
import pandas as pd

#### Parameter experiments


print(sys.argv)
folder = "/Users/rcabanas/GoogleDrive/UAL/causality/dev/bcause/papers/gradient_journal/models/synthetic/s123_original"
target = "/Users/rcabanas/GoogleDrive/UAL/causality/dev/bcause/papers/gradient_journal/models/synthetic/s123"

import shutil

# Delete if target exists
if os.path.exists(target):
    shutil.rmtree(target)

os.makedirs(target)

#folder = sys.argv[1]

listdir = os.listdir(folder)
models = [f.removesuffix("_info.csv") for f in listdir if f.endswith("info.csv") if all(pd.read_csv(Path(folder, f)).markovianity<2) ]

i=0
for m in models:
    mfiles = [f for f in listdir if f.startswith(m)]
    if any([f.endswith("_ccve.csv") for f in  mfiles]):
        i += 1
        for f in mfiles:
            src = Path(folder,f)
            dst = Path(target,f)

            shutil.copyfile(src, dst)
            print(f"copied {f}")

print(f"copied {i} models")