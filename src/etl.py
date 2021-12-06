import json
import os
import sys

sys.path.insert(0, 'src')

import scipy
from scipy import sparse

def read_data(outdir, file):
    fp = os.path.join(outdir, file)
    return scipy.sparse.load_npz(fp)
def get_data_test_adt():
    adt_subset = sparse.load_npz(os.path.abspath(os.getcwd())+"/test/testdata/adt_test_two.npz")
    return adt_subset