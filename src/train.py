from scipy import sparse
import sys
import os
import json


import anndata as ad
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random

import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

import plotly.io as plt_io
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l1
from tensorflow.keras.optimizers import Adam 
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import torch

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse
from model import AE_gex,AE_adt
def get_train_gex(df):
    return AE_gex(input_shape=134)

def get_train_adt(df):
    return AE_adt(input_shape=134)