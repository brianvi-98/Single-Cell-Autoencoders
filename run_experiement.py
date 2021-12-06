#!/usr/bin/env python

import sys
import os
import json

sys.path.insert(0, 'src')

from etl import get_data_test,get_data_train
from model import build_model,train_model,predict_mod
from features import convert_sparse_matrix_to_sparse_tensor
from train import get_train_gex,get_train_adt
def main(targets):
    if 'test' in targets:
        train_data = get_data_train()
        test_data = get_data_test()
        
        torch_train_data = convert_sparse_matrix_to_sparse_tensor(train_data)
        
        
        torch_test_data = convert_sparse_matrix_to_sparse_tensor(test_data)
        mod = train_model(torch_train_data)
        loss_test = predict_mod(mod,torch_test_data).item()
        print("loss of test set: " +str(loss_test))
        return loss_test
    print(get_train_adt('hi'))

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)