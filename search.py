
import torch
import numpy as np

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu") 
# from main import *

# def model_complexity(model):
#   pytorch_total_params = sum(p.numel() for p in model.parameters())
#   print('Total:      \t\t', pytorch_total_params/10**6)
#   pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
#   print('Trainable:\t\t', pytorch_train_params/10**6)
#   return (pytorch_train_params/10**6)

# Validation set estimation
# def val_estimator(model,val_loaer):
#   # with torch.no_grad():
#   #   model.eval()
#   #   val_losses_L1 = []
#   #   for x_val, y_val in val_loader:
#   #       model.eval()
#   #     # x_val, y_val = next(iter(val_loader))
#   #       x_val = x_val.to(device)
#   #       y_val = y_val.to(device)
#   #       y_hat = model(x_val)
#   #       val_loss = loss_measure(y_val,y_hat)   #L1
#   #       val_losses_L1.append(val_loss.item()) 
#   #   print(st.mean(val_losses_L1))
#   #   return val_losses_L1

#def save_csv(val_temp, threshold_temp, Learning_r, Layer_s ):
 # vals_matlab = pd.DataFrame(val_temp)
 # threshold_matlab = pd.DataFrame(threshold_temp)


 # name_vals = 'vals_lr_'+str(Learning_r)+'_layer_size_'+str(Layer_s)+'.csv'
 # name_threshold = 'threshold_lr_'+str(Learning_r)+'_layer_size_'+str(Layer_s)+'.csv'

 # vals_matlab.to_csv(name_vals,index = False,header = False)
 # threshold_matlab.to_csv(name_threshold,index = False,header = False)

 # %cp {name_vals} /content/gdrive/My\ Drive/SiO2_Ag/MODELS/Inverse_Net/
 # %cp {name_threshold} /content/gdrive/My\ Drive/SiO2_Ag/MODELS/Inverse_Net/

# def net_saver(fold, Layer_s, Learning_r, model):
#   results_dir    = 'Results'
#   k_fold_number  = 'Group_Number_' + str(fold)
#   folder_name    =  k_fold_number  + '_lr_' + str(Learning_r) + '_layer_size_'+str(Layer_s)

#   splitting_dir = os.path.join(
#         results_dir, k_fold_number)
#   checkpoint_dir = os.path.join(
#         splitting_dir, folder_name)
#   # name = 'FoldNum_' + str(fold) + '_lr_'+str(Learning_r)+'_layer_size_'+str(Layer_s)+'.pt'
#   # print(name,'  Was saved successfully')
#   # model_save_name = name
#   # path = F"/content/gdrive/My Drive/SiO2_Ag/MODELS/Inverse_Net/Collection/Models/{model_save_name}" 
#   # torch.save(model.state_dict(), path)
#   name = k_fold_number + '_model_lr_'+str(Learning_r) + '_layer_size_'+str(Layer_s)+'.csv'
#   name = os.path.join(checkpoint_dir, '_model.pt')

# def net_loader(fold, Layer_s,Learning_r):
#   name = 'FoldNum_' + str(fold) + '_lr_'+str(Learning_r)+'_layer_size_'+str(Layer_s)+'.pt'
#   model_save_name = name
#   testm = NN(Layer_s).to(device)
#   path = F"/content/gdrive/My Drive/SiO2_Ag/MODELS/Inverse_Net/Collection/Models/{model_save_name}"
#   testm.load_state_dict(torch.load(path))
#   print(name,'  Was loaded successfully')
#   return testm 

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
  # val_temp = -999*np.ones((300,5))
  # threshold_temp= +1000*np.ones((300,5))

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

#Loading
# def net_loader(Layer_s,Learning_r):
#   name = 'lr_'+str(Learning_r)+'_layer_size_'+str(Layer_s)+'.pt'
#   model_save_name = name
#   testm = NN(Layer_s).to(device)
#   path = F"/content/gdrive/My Drive/SiO2_Ag/MODELS/Inverse_Net/{model_save_name}"
#   testm.load_state_dict(torch.load(path))
#   print(name,'  Was loaded successfully')
#   return testm