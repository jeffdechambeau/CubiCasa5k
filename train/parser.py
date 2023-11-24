
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use.')
    parser.add_argument('--optimizer', nargs='?', type=str, default='adam-patience-previous-best',
                        help='Optimizer to use [\'adam, sgd\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of the epochs')
    parser.add_argument('--n-epoch', nargs='?', type=int, default=1000,
                        help='# of the epochs')
    parser.add_argument('--batch-size', nargs='?', type=int, default=26,
                        help='Batch Size')
    parser.add_argument('--image-size', nargs='?', type=int, default=256,
                        help='Image size in training')
    parser.add_argument('--l-rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--l-rate-var', nargs='?', type=float, default=1e-3,
                        help='Learning Rate for Variance')
    parser.add_argument('--l-rate-drop', nargs='?', type=float, default=200,
                        help='Learning rate drop after how many epochs?')
    parser.add_argument('--patience', nargs='?', type=int, default=10,
                        help='Learning rate drop patience')
    parser.add_argument('--feature-scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file .pkl')
    parser.add_argument('--furukawa-weights', nargs='?', type=str, default=None,
                        help='Path to previously trained furukawa model weights file .pkl')
    parser.add_argument('--new-hyperparams', nargs='?', type=bool,
                        default=False, const=True,
                        help='Continue training with new hyperparameters')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')
    parser.add_argument('--debug', nargs='?', type=bool,
                        default=False, const=True,
                        help='Continue training with new hyperparameters')
    parser.add_argument('--plot-samples', nargs='?', type=bool,
                        default=False, const=True,
                        help='Plot floorplan segmentations to Tensorboard.')
    parser.add_argument('--scale', nargs='?', type=bool,
                        default=False, const=True,
                        help='Rescale to 256x256 augmentation.')
    args = parser.parse_args()
    return args
