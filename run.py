#!/usr/bin/env python

import sys
import os
import json

sys.path.insert(0, 'src')

from etl import read_data,get_data_test_adt
from model import predict_mod
from features import convert_sparse_matrix_to_sparse_tensor
from train import get_train_gex, get_train_adt
import torch



def main(targets):
    adt_model = None
    gex_model = None
    if 'test' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
         
        data_gex = read_data(**data_cfg, file='train_data_gex.npz')
        data_adt = read_data(**data_cfg, file='train_data_adt.npz')
    
        torch_gex_data = convert_sparse_matrix_to_sparse_tensor(data_gex).to(device)
        torch_adt_data = convert_sparse_matrix_to_sparse_tensor(data_adt).to(device)
            
        #if gex_model == None:
            #gex_model = get_train_gex(torch_gex_data)
            
        if adt_model== None:
            adt_model = get_train_adt(torch_adt_data)
        

        test_data_adt= get_data_test_adt()
        
        
        torch_test_data_adt = convert_sparse_matrix_to_sparse_tensor(test_data_adt)
        loss_test = predict_mod(adt_model,torch_test_data_adt).item()
        print("loss of test set: " +str(loss_test))
        return loss_test
    print('hi')

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)