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
         # if part >=3 and part <5:
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
    # Feel free to add more argument parameters
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
                        help='Seed to use for reproducing results')
    parser.add_argument('--n_splits', default = 10, type = int,
                        help = 'Number of partitions for training in K-Fold cross-validations')
    parser.add_argument('--parts', type = list, default = np.arange(1,11),
                        help = 'Partitions which we want consider. By default: all 10 partitions. See "n_splits"')


    # parser.add_argument('--hidden_dims_gen', default=[128, 256, 512],
    #                     type=int, nargs='+',
    #                     help='Hidden dimensionalities to use inside the ' +
    #                          'generator. To specify multiple, use " " to ' +
    #                          'separate them. Example: \"128 256 512\"')
    # parser.add_argument('--hidden_dims_disc', default=[512, 256],
    #                     type=int, nargs='+',
    #                     help='Hidden dimensionalities to use inside the ' +
    #                          'discriminator. To specify multiple, use " " to ' +
    #                          'separate them. Example: \"512 256\"')
    # parser.add_argument('--dp_rate_gen', default=0.1, type=float,
    #                     help='Dropout rate in the discriminator')
    # parser.add_argument('--dp_rate_disc', default=0.3, type=float,
    #                     help='Dropout rate in the discriminator')

    # # Optimizer hyperparameters
    # parser.add_argument('--lr', default=2e-4, type=float,
    #                     help='Learning rate to use')
    # parser.add_argument('--batch_size', default=128, type=int,
    #                     help='Batch size to use for training')

    # # Other hyperparameters
    
    # parser.add_argument('--seed', default=42, type=int,
    #                     help='Seed to use for reproducing results')

    # parser.add_argument('--num_workers', default=4, type=int,
    #                     help='Number of workers to use in the data loaders.' +
    #                          'To have a truly deterministic run, this has to be 0.')
    # parser.add_argument('--log_dir', default='GAN_logs/', type=str,
    #                     help='Directory where the PyTorch Lightning logs ' +
    #                          'should be created.')
    # parser.add_argument('--progress_bar', action='store_true',
    #                     help='Use a progress bar indicator for interactive experimentation. ' +
    #                          'Not to be used in conjuction with SLURM jobs.')

    # # saving images
    # parser.add_argument('--save_every', default=10, type=int,
    #                     help='Number of epochs to generate an image.')

    args = parser.parse_args()

    main(args)  