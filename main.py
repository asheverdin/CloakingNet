from training_methods import train_main
from dataimport import *
from sklearn.model_selection import KFold


sys.path.append("/.")
np.random.seed(17)
torch.manual_seed(17)

#splitting into 10 partitions
kf = KFold(n_splits=10,random_state=256, shuffle=True)
kf.get_n_splits(X)

print(kf)  
part = 0
for train_index, test_index in kf.split(X):
       part += 1
       if part >=3 and part <5:
          #object that stores data and saves/loads net parameters
          print('Part number:\t', part)
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = y[train_index], y[test_index]

          x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor, = \
          torch.from_numpy(X_train).float(),\
          torch.from_numpy(X_test).float(),\
          torch.from_numpy(y_train).float(),\
          torch.from_numpy(y_test).float()  
    #Loader
          train_loader, train_loader_check, test_loader = loader_set(64, 'whole', x_train_tensor,\
                                                                      y_train_tensor, x_test_tensor,\
                                                                      y_test_tensor)
    # Training
          train_main(part, train_loader, train_loader_check, test_loader)
    #Model   
  