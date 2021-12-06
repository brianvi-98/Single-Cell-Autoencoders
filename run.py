import sys
import os
import json

sys.path.insert(0, 'src')

from etl import read_data
from models import to_torch, create_AE

def main(targets):
    if 'test' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        
        data_gex = read_data(**data_cfg, file='test_data_gex.npz')
        data_adt = read_data(**data_cfg, file='test_data_adt.npz')
        
        gex_torch_data = to_torch(data_gex)
        adt_torch_data = to_torch(data_adt)
        
        gex_autoencoder, adt_autoencoder = create_AE()
        
        gex_outputs = gex_autoencoder(gex_torch_data)
        adt_outputs = adt_autoencoder(adt_torch_data)
    
    return gex_outputs, adt_outputs
        
    
        
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)