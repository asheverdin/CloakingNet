
import torch
import numpy as np

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu") 

def decrease(bestmodel):
  bestmodel.eval()
  expected = torch.ones(1,5)
  threshold = 9
  j=0
  vals_arr = []
  threshold_arr = []

  # (expected<0).any()== False and 
  while   threshold> -2: 
    vals = expected
    # print((vals.cpu().data, threshold))
    threshold = (0.1-0.0025*j)
      

    true_x_np = threshold *np.ones((1,151))
    true_x = torch.from_numpy( true_x_np.reshape(1,151) ).float()
    expected = bestmodel(true_x.to(device)).view(1,-1)
    vals_arr.append(expected.cpu().data.numpy())
    threshold_arr.append(threshold)
    
    j+=1
  # print('iter',j, '\tthreshold', threshold,'\tpredict', vals )

  vals_arr = np.array(vals_arr)
  vals_arr = vals_arr.reshape(vals_arr.shape[0],-1)

  threshold_arr = np.array(threshold_arr)
  # vals_arr = vals_arr.reshape(vals_arr.shape[0],-1)

  print(vals_arr.shape)
  print(threshold_arr.shape)
  return vals_arr, threshold_arr 



def extract_positive(vals_arr, threshold_arr): 

  val_temp = []
  threshold_temp= []  
  counter = 0;

  main_val_matlab = []
  main_tr_matlab = []
  for i,row in enumerate(vals_arr):
      if (row<1).any() == False:
        # val_temp[counter] = row
        # print(row)
        # threshold_temp[counter] = threshold_arr[i]
        val_temp.append(row)
        threshold_temp.append( threshold_arr[i])
        counter +=1


  print(counter)
  val_temp = np.array(val_temp)
  val_temp.shape

  threshold_temp = np.array(threshold_temp)
  threshold_temp.shape
  
  return  val_temp, threshold_temp

