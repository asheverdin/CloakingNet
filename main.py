from training_methods import train_main
from dataimport import *
from sklearn.model_selection import KFold
import argparse


def main(args):
  print(args)
  sys.path.append("/.")
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  #splitting into 10 partitions
  kf = KFold(n_splits=args.n_splits, random_state=256, shuffle=True)
  kf.get_n_splits(X)
  device = args.device
  print(kf)  
  part = 0

  for train_index, test_index in kf.split(X):
         part += 1
         # which partitions to train on: All of them by default
         if part in args.parts:
            #object that stores data and saves/loads net parameters
            print('Part number:\t', part)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor, = \
            torch.from_numpy(X_train).float().to(device),\
            torch.from_numpy(X_test).float().to(device),\
            torch.from_numpy(y_train).float().to(device),\
            torch.from_numpy(y_test).float().to(device)  
      #Loader
            train_loader, train_loader_check, test_loader = loader_set(args.batch_size, 'whole', x_train_tensor,\
                                                                        y_train_tensor, x_test_tensor,\
                                                                        y_test_tensor)
      # Training
            train_main(part, train_loader, train_loader_check, test_loader, args)
      #Model   
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--epochs', default = 300, type = int,
                        help='Number of epochs for training')
    parser.add_argument('--stopping_criterion', default=20, type = int,
                        help='N iterations of non-improving validation score until early stopping')
    parser.add_argument('--batch_size', default = 32, type = int,
                        help='Batch size used for training')
    parser.add_argument('--device', type = str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device: CPU or GPU on which training is conducted")
    parser.add_argument('--seed', default = 17, type=int,
                        help='Seed to reproduce the results')
    parser.add_argument('--n_splits', default = 10, type = int,
                        help = 'Number of pars for training in K-Fold cross-validations')
    parser.add_argument('--parts', type = list, default = np.arange(1,11),
                        help = 'Parts which we want consider. By default: all 10 parts of partition. See "n_splits"')


    args = parser.parse_args()

    main(args)  