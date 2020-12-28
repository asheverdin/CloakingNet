import os
from model import NN
import torch
import pandas as pd

class DataStore():

  def __init__(self, part, Learning_r, Layer_s ):
      """
      Class that trains, saves the predicted dimesnions and saves/loads the network states.
      Inputs:
          part - number of the part in k-fold splitting.
          Learning_r - learning rate of the Neural Network
          Layer_s    - number of nodes in each of all of the layers
      """
      super().__init__()
      #Values that are later passed to MATLAB for faster computations
      self.part = part
      self.Learning_r = Learning_r
      self.Layer_s    = Layer_s
      # Folders
      self.results_dir  = 'Results'
      self.part_number  = 'Part_Number_' + str(self.part)
      self.folder_name  =  self.part_number  + '_lr_' + str(self.Learning_r) + '_layer_size_'+str(self.Layer_s)

    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
      self.part_dir = os.path.join(
        self.results_dir, self.part_number)
      self.checkpoint_dir = os.path.join(
        self.part_dir, self.folder_name)

      os.makedirs(self.part_dir,  exist_ok = True)
      os.makedirs(self.checkpoint_dir, exist_ok = True)
        
  def net_saver(self, model_to_save):
      model_name = self.part_number + '_model_lr_' + str(self.Learning_r) + '_layer_size_' + str(self.Layer_s) + '.pt'
      model_path = os.path.join(self.checkpoint_dir, model_name)
      torch.save(model_to_save.state_dict(), model_path)
      print(model_name, 'Was saved successfully \t\t\t[saved]')

  def net_loader(self, path = None):
      testm = NN(self.Layer_s).to(device)
      if path is None:
        model_name = self.part_number + '_model_lr_' + str(self.Learning_r) + '_layer_size_' + str(self.Layer_s) + '.pt'
        path = os.path.join(self.checkpoint_dir, model_name)
        testm.load_state_dict(torch.load(path))
        print(model_name,' Was loaded successfully loaded.\t\t\t [loaded]')
      else:
        testm.load_state_dict(torch.load(path))
        print(model_name,' Was loaded successfully loaded from the path.\t\t\t [loaded from Path]')
      return testm  

  def records_saver(self, records):
      self.records  = records
      self.name_records =  'Part_Number_' + str(self.part) + '_records.csv'
      self.records.to_csv(os.path.join(self.checkpoint_dir, name_records),index = True,header = True)

  def dimensions_saver(self, d_values, threshold):
      """
      Method that saves the predicted dimesnion and thresholds of constant scattering values
      Inputs:
          split - number of  the split in k-fold splitting.
          Learning_r - learning rate of the Neural Network
          Layer_s    - number of nodes in each of all of the layers
      """
      #used in MATLAB 
      self.vals_matlab = pd.DataFrame(d_values)
      self.threshold_matlab = pd.DataFrame(threshold)

      #Summary of the results
      name_vals = self.part_number  + '_vals_lr_'+ str(self.Learning_r) + '_layer_size_'+str(self.Layer_s)+'.csv'
      name_threshold = self.part_number  + '_threshold_lr_'+str(self.Learning_r) + '_layer_size_'+str(self.Layer_s)+'.csv'

      #saving d_values, corresponding thresholds
      self.vals_matlab.to_csv(os.path.join(self.checkpoint_dir, name_vals), index = False, header = False)
      self.threshold_matlab.to_csv(os.path.join(self.checkpoint_dir, name_threshold), index = False, header = False)
    

# def train_main(part, train_loader, train_loader_check, test_loader):
#     """
#       Conducts main training procedure for particular part of k-fold splitting.
#     """
#     records = pd.DataFrame(columns=['Part_Num','Learning_rate', 'Layer_size', 'best_val(MSE)',\
#                                     'last_val(MARE)', 'stopping epoch', 'counter',\
#                                     'positive num', 'all'])
#     # sets of learning rates and numbers of nodes in layers.
#     lr_arr = np.array([0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]).reshape(-1,1)
#     N_size = np.array([100,    200,    300,  400,  500,  600]).reshape(-1,1)

#     for i, LR in enumerate(lr_arr):
#       for j, LS in enumerate(N_size):
#         Learning_rate = LR[0]
#         Layer_size    = LS[0]

#     # Data storing object
#         ds  = DataStore(part, Learning_rate, Layer_size)
#     # Initialize
#         loss_main =  nn.MSELoss()
#         loss_secondaty = MARE
#         model, optimizer, train_step, counter = params_init(Layer_size, Learning_rate, loss_main) #parameters
#         complexity = model_complexity(model)   
#         print("%-10s\n %-15s %-4.6f \n %-15s %-4.2f\n %10s "%("////////////////////////",\
#                                                        "Learning Rate:", Learning_rate,\
#                                                        "Layer Size:",    Layer_size,\
#                                                        "////////////////////////"))
#     # #Train                                       #number of trainable parameters
#         bestmodel, best_val_error_main, best_val_error_secondary, counter, epoch = train_procedure(model, \
#                                                               train_step,loss_main,
#                                                               loss_secondaty, \
#                                                               train_loader, train_loader_check, \
#                                                               test_loader)#training
#     #save the model
#         ds.net_saver(bestmodel)
#     # "Scanning" over the constant(threshold) scattering power values.
#         vals_arr, threshold_arr =  decrease(bestmodel)
#         val_temp, threshold_temp = extract_positive(vals_arr, threshold_arr)
#     # save thresholds and dimensions' values of a "scan" in the form: [d1, d2, d3, d4, d5]
#         ds.dimensions_saver(val_temp, threshold_temp)

#         new_row = {'Part_Num': part,
#                   'Learning_rate': Learning_rate, \
#                   'Layer_size': Layer_size,\
#                   'best_val(MSE)': best_val_error_main,\
#                   'last_val(MARE)': best_val_error_secondary,
#                   'stopping epoch': epoch, \
#                   'counter': counter,\
#                   'positive num':val_temp.shape[0] ,\
#                   'all': vals_arr.shape[0]}
#         records = records.append(new_row, ignore_index=True) 
#     ds.records_saver(records) 
