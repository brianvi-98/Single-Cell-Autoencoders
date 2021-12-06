import json
import os
import sys

sys.path.insert(0, 'src')

import scipy
from scipy import sparse

def read_data(outdir, file):
    fp = os.path.join(outdir, file)
    return scipy.sparse.load_npz(fp)
