
import pandas as pd
import numpy as np
import torch 
import sys
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

sys.path.append("/.")

# testing sample
df_x_test = pd.read_csv('./Data/x_test.csv', header=None);
df_y_test = pd.read_csv('./Data/y_test.csv', header=None);

# training - validation
df_x_small = pd.read_csv('./Data/x_train_val.csv', header=None);
df_y_small = pd.read_csv('./Data/y_train_val.csv', header=None);

X = np.array(df_x_small)
y = np.array(df_y_small)

# ALL:  NUMPY -> TENSOR
x_tensor    = torch.from_numpy( X ).float()
y_tensor    = torch.from_numpy( y ).float()

x_test_tensor =   torch.from_numpy( np.array(df_x_test) ).float()
y_test_tensor =   torch.from_numpy( np.array(df_y_test) ).float()


def loader_set(train_batch_size,test_batch_size, x_train, y_train, x_test, y_test ):
  # Warning! x and y are flipped  (x,y) -> (y,x). Batch passes as scattering values as inputs
  # the y - values
  
  if test_batch_size == 'whole': 
     test_batch_size = x_test.shape[0]

  train_dataset = TensorDataset( y_train, x_train)
  test_dataset =  TensorDataset( y_test, x_test)
 
  
  train_loader = DataLoader(dataset=train_dataset,       batch_size=train_batch_size,                        shuffle=True)
  train_loader_check = DataLoader(dataset=train_dataset, batch_size = x_train.shape[0],                      shuffle=True)
  val_loader   = DataLoader(dataset=test_dataset,         batch_size = test_batch_size ,                      shuffle=True)
  # test_loader  = DataLoader(dataset=test_dataset,        batch_size = x_test_tensor.shape[0],          shuffle=True)

  return train_loader, train_loader_check, val_loader
