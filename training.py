from operator import mod
import os
import argparse
from model import Model


if (__name__ == '__main__'):

    # Parse the execution arguments
    parser = argparse.ArgumentParser(description='Train neural network')

    # Add the arguments that we need the user to provide (if not, it will default them)
    parser.add_argument('--conf', '--conf', default='conf.dat', type=str, metavar='CONF', help='Configuration file')
    parser.add_argument('--gpu', '--gpu', default=0, type=int, metavar='GPU', help='GPU')
    parser.add_argument('--batch', '--batch', default=128, type=int, metavar='BATCH', help='Batch size')
    parser.add_argument('--split', '--split', default=0.9, type=float, metavar='SPLIT', help='Training split')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float, metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=2500, type=int, metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--rd', '--readir', default=f'data/', metavar='READIR', help='directory for reading the training data')
    parser.add_argument('--sav', '--savedir', default=f'checkpoints/test/', metavar='SAVEDIR', help='directory for output files')

    # convert the arguments to an object
    parsed = vars(parser.parse_args())

    # If the save directory does not exist, create it
    if not os.path.exists(parsed['sav']):
        os.makedirs(parsed['sav'])

    # create the model
    model = Model(parsed['conf'], parsed['gpu'])

    # train the model
    model.train(parsed['epochs'], parsed['lr'], parsed['batch'], parsed['split'], parsed['smooth'], parsed['rd'], parsed['sav'])
    model.summary()