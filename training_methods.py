
from model import *
from data_store import DataStore
from helper import *
from search import *

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import statistics as st
import copy



def MARE(input,target):
  #Mean Average Relative Error or relative accuracy
  loss =  torch.mean( torch.abs (input - target)/target)
  return loss

def params_init(layer_size, learning_r, loss_main):
# model setting    
  model1 = NN(layer_size).to(device)
  model1.apply(init_weights) 

  optimizer1 = torch.optim.Adam(model1.parameters(), lr= learning_r) 
  train_step1 = make_train_step(model1, loss_main, optimizer1)
  counter1 = 0
  return model1, optimizer1, train_step1, counter1

def train_procedure(model, train_step,loss_main,loss_secondary,
                  train_loader, train_loader_check, val_loader, args):

  n_epochs = args.epochs
  device = args.device
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  counter = 0
  i=0
  k = 0
#   tr_losses_main =        []
#   val_losses_main =       []
#   tr_losses_secondary =   []
#   val_losses_secondary=   []
  
  tr_losses_main_global =       []
  val_losses_main_global =      []
  tr_losses_secondary_global =  []
  val_losses_secondary_global = []
  
  best_val_error_main =     999
  final_error_secondary =   999

  for epoch in range(0, n_epochs):
      display = 0
      tr_losses_main =  []
      tr_losses_secondary= []

      val_losses_main = []
      val_losses_secondary = []    
        
# train step is made        
      for x_batch, y_batch in train_loader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)
          loss = train_step(x_batch, y_batch)
          tr_losses_main.append(loss)                                               #
# evaluation stage
      with torch.no_grad():
          model.eval()
# validation step is made
          for x_val, y_val in val_loader:
              x_val = x_val.to(device)
              y_val = y_val.to(device)
              y_hat = model(x_val)
              val_losses_main_temp      = loss_main(y_hat,y_val)   #main
              val_losses_secondary_temp = loss_secondary(y_hat,y_val) #secondary
#appends val_losses(main+secondary)
              val_losses_main.append(val_losses_main_temp.item()) #Appends loss     #
              val_losses_secondary.append(val_losses_secondary_temp.item())         #
              
          for x_tr, y_tr in train_loader_check:
              x_tr = x_tr.to(device)
              y_tr = y_tr.to(device)
              y_hat = model(x_tr)
              tr_losses_secondary_temp = loss_secondary(y_hat, y_tr)
              tr_losses_secondary.append(tr_losses_secondary_temp.item())           #

          counter += 1
          if (st.mean(val_losses_main) < best_val_error_main):
              state_dict_best_1 = model.state_dict()
              best_val_error_main = st.mean(val_losses_main)
              final_error_secondary = st.mean(val_losses_secondary)
              
              bestm = copy.deepcopy(model)
              counter = 1
          
          MSE_TR  = st.mean(tr_losses_main)
          MSE_VL  = st.mean(val_losses_main)
          MARE_TR = st.mean(tr_losses_secondary)
          MARE_VL = st.mean(val_losses_secondary)
          if  counter >= 1:
                    print("%-5s %-i\t %-5s %-i  %-10s\t %-4.4f\t  %-10s\t %-4.4f\t %-10s\t %-4.4f\t %-10s\t %-4.4f \n" % \
                                       ( "Epochs", epoch+1, "Counter", counter,\
                                        "[T]train(MSE):",            np.round(MSE_TR,4),\
                                        "[E]val(MSE),%:",            np.round(MSE_VL,4),
                                        "[E]train(MRAE),%:",   100 * np.round( MARE_TR,4),\
                                        "[E]val(MRAE),%:",     100 * np.round(MARE_VL,4)))
          tr_losses_main_global.append(MSE_TR)
          val_losses_main_global.append(MSE_VL)
          tr_losses_secondary_global.append(MARE_TR)
          val_losses_secondary_global.append(MARE_VL)
               
          if counter == args.stopping_criterion:
            print('//////////////////////////////////The End /////////////////////////////////////////////////////////////////////////////////////////////////')
            break;
  return bestm, best_val_error_main,final_error_secondary, counter, epoch+1


def train_main(part, train_loader, train_loader_check, test_loader, args):
    """
      Conducts main training procedure for particular part of k-fold splitting.
      *Iterates over the set of hyperparameters.
      *After training "scans" all the thresholds, generating the dimension values [d1,d2,d3,d4,d5] of particles. 
      *Calls an object of DataStore class to save the results and the best network states.
    """
    device = args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = pd.DataFrame(columns=['Part_Num','Learning_rate', 'Layer_size', 'best_val(MSE)',\
                                    'last_val(MARE)', 'stopping epoch', 'counter',\
                                    'positive num', 'all'])
    # sets of learning rates and numbers of nodes in layers.
    lr_arr = np.array([0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]).reshape(-1,1)
    N_size = np.array([100,    200,    300,  400,  500,  600]).reshape(-1,1)

    for i, LR in enumerate(lr_arr):
      for j, LS in enumerate(N_size):
        Learning_rate = LR[0]
        Layer_size    = LS[0]

    # Data storing object
        ds  = DataStore(part, Learning_rate, Layer_size)
    # Initialize
        loss_main =  nn.MSELoss()
        loss_secondaty = MARE
        model, optimizer, train_step, counter = params_init(Layer_size, Learning_rate, loss_main) #parameters
        complexity = model_complexity(model)   
        print("%-10s\n %-15s %-4.6f \n %-15s %-4.2f\n %10s "%("////////////////////////",\
                                                       "Learning Rate:", Learning_rate,\
                                                       "Layer Size:",    Layer_size,\
                                                       "////////////////////////"))
    # #Train                                       #number of trainable parameters
        bestmodel, best_val_error_main, best_val_error_secondary, counter, epoch = train_procedure(model, \
                                                              train_step,loss_main,
                                                              loss_secondaty, \
                                                              train_loader, train_loader_check, \
                                                              test_loader,\
                                                              args)#training
    #save the model
        ds.net_saver(bestmodel)
    # "Scanning" over the constant(threshold) scattering power values.
        vals_arr, threshold_arr =  decrease(bestmodel)
        val_temp, threshold_temp = extract_positive(vals_arr, threshold_arr)
    # save thresholds and dimensions' values of a "scan" in the form: [d1, d2, d3, d4, d5]
        ds.dimensions_saver(val_temp, threshold_temp)

        new_row = {'Part_Num': part,
                  'Learning_rate': Learning_rate, \
                  'Layer_size': Layer_size,\
                  'best_val(MSE)': best_val_error_main,\
                  'last_val(MARE)': best_val_error_secondary,
                  'stopping epoch': epoch, \
                  'counter': counter,\
                  'positive num':val_temp.shape[0] ,\
                  'all': vals_arr.shape[0]}
        records = records.append(new_row, ignore_index=True) 
    ds.records_saver(records) 

  