#!/usr/bin/env python

import sys
import os
import json

sys.path.insert(0, 'src')

from etl import read_data
from model import build_model, train_model, predict_mod
from features import convert_sparse_matrix_to_sparse_tensor
from train import get_train_gex, get_train_adt

adt_model = None
gex_model = None
with open('config/train-params.json') as fh:
    train_cfg = json.load(fh)

epoch_gex = train_cfg['epoch_count_gex']
epoch_adt = train_cfg['epoch_count_adt']
batch_size = train_cfg['batch_size']
def main(targets):
    if 'test' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
         
        data_gex = read_data(**data_cfg, file='train_data_gex.npz')
        data_adt = read_data(**data_cfg, file='train_data_adt.npz')
        
        torch_gex_data = convert_sparse_matrix_to_sparse_tensor(data_gex)
        torch_adt_data = convert_sparse_matrix_to_sparse_tensor(data_adt)
        train_data = get_train_adt(torch_adt_data)
        #test_data = get_data_test()
        
        torch_train_data = convert_sparse_matrix_to_sparse_tensor(train_data)
        
        
        torch_test_data = convert_sparse_matrix_to_sparse_tensor(test_data)
        mod = train_model(torch_train_data)
        loss_test = predict_mod(mod,torch_test_data).item()
        print("loss of test set: " +str(loss_test))
        return loss_test
    print('hi')

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)