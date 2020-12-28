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
