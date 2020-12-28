
import torch
from torch import nn
class NN(nn.Module):
    def __init__(self, layer_size ):
        super(NN, self).__init__()
        self.main = nn.Sequential(
#              nn.BatchNorm1d(151),
             nn.Linear(151,layer_size),
             nn.LeakyReLU(0.3),
#              nn.BatchNorm1d(self.fc_size),
#              nn.Dropout(p=0.5),
             nn.Linear(layer_size,layer_size),
             nn.LeakyReLU(0.3),
#              nn.BatchNorm1d(self.fc_size),
#              nn.Dropout(p=0.5),
             nn.Linear(layer_size,layer_size),
             nn.LeakyReLU(0.3),
#              nn.Dropout(p=0.5),
#              nn.BatchNorm1d(self.fc_size),
             nn.Linear(layer_size,5),
#              nn.LeakyReLU(0.3),
#              nn.BatchNorm1d(5),
        )
        
    def forward(self, input):
         return self.main(input)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def model_complexity(model):
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  print('Total:      \t\t', pytorch_total_params/10**6)
  pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
  print('Trainable:\t\t', pytorch_train_params/10**6)
  return (pytorch_train_params/10**6)