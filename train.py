import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision.transforms import RandomChoice
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from floortrans.metrics import get_px_acc, runningScore
from floortrans.losses import UncertaintyLoss
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import (
    RandomCropToSizeTorch, ResizePaddedTorch, Compose, DictToTensor,                      ColorJitterTorch, RandomRotations)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')


def setup_augmentation(args):
    augmentations = [RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                     RandomRotations(format='cubi'),
                     DictToTensor(),
                     ColorJitterTorch()]

    if args.scale:
        augmentations.insert(0, RandomChoice([ResizePaddedTorch(
            (0, 0), data_format='dict',                                        size=(args.image_size, args.image_size))]))
    return Compose(augmentations)


def setup_dataloader(args, aug):
    train_set = FloorplanSVG(args.data_path, 'train.txt',
                             format='lmdb', augmentations=aug)
    val_set = FloorplanSVG(args.data_path, 'val.txt',
                           format='lmdb', augmentations=DictToTensor())

    num_workers = 0 if args.debug else 8
    trainloader = data.DataLoader(train_set, batch_size=args.batch_size,
                                  num_workers=num_workers,          shuffle=True, pin_memory=True)
    valloader = data.DataLoader(
        val_set, batch_size=1, num_workers=num_workers, pin_memory=True)
    return trainloader, valloader


def setup_model(args, logger):
    model, criterion = get_model_and_criterion(args, logger)
    model.cuda()
    return model, criterion


def get_model_and_criterion(args, logger):
    input_slice = [21, 12, 11]
    if args.arch == 'hg_furukawa_original':
        model, criterion = setup_furukawa_model(args, logger, input_slice)
    else:
        model = get_model(args.arch, args.n_classes)
        criterion = UncertaintyLoss(input_slice=input_slice)
    return model, criterion


def setup_furukawa_model(args, logger, input_slice):
    model = get_model(args.arch, 51)
    criterion = UncertaintyLoss(input_slice=input_slice)
    # Additional setup specific to Furukawa's architecture...
    return model, criterion


def train_epoch(args, model, criterion, optimizer, trainloader, writer, epoch):
    model.train()
    total_loss = 0.0
    for i, samples in tqdm(enumerate(trainloader), total=len(trainloader), ncols=80, leave=False):
        images = samples['image'].cuda(non_blocking=True)
        labels = samples['label'].cuda(non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(trainloader)
    writer.add_scalar('training/loss', avg_loss, epoch)
    logging.info(f"Epoch [{epoch+1}/{args.n_epoch}] Loss: {avg_loss:.4f}")


def validate_epoch(args, model, criterion, valloader, writer, epoch):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for i_val, samples_val in tqdm(enumerate(valloader), total=len(valloader), ncols=80, leave=False):
            images_val = samples_val['image'].cuda(non_blocking=True)
            labels_val = samples_val['label'].cuda(non_blocking=True)

            outputs = model(images_val)
            loss = criterion(outputs, labels_val)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(valloader)
    writer.add_scalar('validation/loss', avg_val_loss, epoch)
    logging.info(f"Validation Loss: {avg_val_loss:.4f}")

    return avg_val_loss


def update_best_metrics_and_save_model(args, avg_val_loss, best_metrics, model, criterion, optimizer, epoch, log_dir, writer, logger):
    if avg_val_loss < best_metrics['best_loss']:
        best_metrics['best_loss'] = avg_val_loss
        logger.info("New best validation loss, saving model...")
        save_model_state(epoch, model, criterion, optimizer,
                         log_dir, "model_best_val_loss.pkl")


def save_model_state(epoch, model, criterion, optimizer, log_dir, filename):
    state = {
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'criterion_state': criterion.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(state, os.path.join(log_dir, filename))


def load_model_if_exists(args, model, criterion, optimizer, logger, log_dir):
    start_epoch = 0

    if args.weights:
        checkpoint_path = os.path.join(log_dir, args.weights)
        if os.path.isfile(checkpoint_path):
            logger.info(
                f"Loading model and optimizer from checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint['model_state'])
            criterion.load_state_dict(checkpoint['criterion_state'])
            if 'optimizer_state' in checkpoint and not args.new_hyperparams:
                optimizer.load_state_dict(checkpoint['optimizer_state'])

            start_epoch = checkpoint.get('epoch', 0)
            logger.info(
                f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
        else:
            logger.info(f"No checkpoint found at '{checkpoint_path}'")
    else:
        logger.info(
            "No weights argument provided, starting training from scratch.")

    return start_epoch


def perform_training(args, model, criterion, optimizer, scheduler, trainloader, valloader, writer, logger, log_dir):
    best_metrics = initialize_best_metrics()
    start_epoch = load_model_if_exists(
        args, model, criterion, optimizer, logger, log_dir)

    for epoch in range(start_epoch, args.n_epoch):
        train_epoch(args, model, criterion, optimizer,
                    trainloader, writer, epoch)
        val_metrics = validate_epoch(
            args, model, criterion, valloader, writer, epoch)

        update_best_metrics_and_save_model(
            args, val_metrics, best_metrics, model, criterion, optimizer, epoch, log_dir, writer, logger)


def initialize_best_metrics():
    return {'best_loss': np.inf, 'best_acc': 0, 'best_train_loss': np.inf, 'best_loss_var': np.inf}


def setup_optimizer_scheduler(args, model, criterion):
    # Combine parameters from model and criterion
    params = [{'params': model.parameters(), 'lr': args.l_rate},
              {'params': criterion.parameters(), 'lr': args.l_rate}]

    # Selecting the optimizer
    if args.optimizer == 'adam-patience' or args.optimizer == 'adam-patience-previous-best':
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, momentum=0.9, weight_decay=10**-4, nesterov=True)
    # Add other optimizers here if needed

    # Setting up the scheduler
    scheduler = None
    if 'adam-patience' in args.optimizer:
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', patience=args.patience, factor=0.5)
    elif args.optimizer == 'adam-scheduler':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 0.5 ** np.floor(epoch / args.l_rate_drop))
    elif args.optimizer == 'sgd':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (1 - epoch / args.n_epoch) ** 0.9)

    return optimizer, scheduler


def train(args, log_dir, writer, logger):
    # Save arguments to a file
    save_args_to_file(args, log_dir)

    # Setup Augmentation, Dataloaders, Model, and Criterion
    aug = setup_augmentation(args)
    trainloader, valloader = setup_dataloader(args, aug)
    model, criterion = setup_model(args, logger)

    # Add parameters to TensorBoard
    writer.add_text('parameters', str(vars(args)))
    log_data_loading(logger)

    # Setup Optimizer and Scheduler
    optimizer, scheduler = setup_optimizer_scheduler(args, model, criterion)

    # Training and Validation Loop
    perform_training(args, model, criterion, optimizer, scheduler,
                     trainloader, valloader, writer, logger, log_dir)


def save_args_to_file(args, log_dir):
    with open(os.path.join(log_dir, 'args.json'), 'w') as out:
        json.dump(vars(args), out, indent=4)


def log_data_loading(logger):
    logger.info('Loading data...')


def parse_arguments():

    arg_specs = [
        ('--arch', {'nargs': '?', 'type': str,
         'default': 'hg_furukawa_original', 'help': 'Architecture to use.'}),
        ('--optimizer', {'nargs': '?', 'type': str, 'default': 'adam-patience-previous-best',
         'help': "Optimizer to use ['adam, sgd']."}),
        ('--data-path', {'nargs': '?', 'type': str,
         'default': 'data/cubicasa5k/', 'help': 'Path to data directory'}),
        ('--n-classes', {'nargs': '?', 'type': int,
         'default': 44, 'help': 'Number of classes.'}),
        ('--n-epoch', {'nargs': '?', 'type': int,
         'default': 1000, 'help': 'Number of epochs'}),
        ('--batch-size', {'nargs': '?', 'type': int,
         'default': 26, 'help': 'Batch Size'}),
        ('--image-size', {'nargs': '?', 'type': int,
         'default': 256, 'help': 'Image size in training'}),
        ('--l-rate', {'nargs': '?', 'type': float,
         'default': 1e-3, 'help': 'Learning Rate'}),
        ('--l-rate-var', {'nargs': '?', 'type': float,
         'default': 1e-3, 'help': 'Learning Rate for Variance'}),
        ('--l-rate-drop', {'nargs': '?', 'type': float, 'default': 200,
         'help': 'Learning rate drop after how many epochs?'}),
        ('--patience', {'nargs': '?', 'type': int,
         'default': 10, 'help': 'Learning rate drop patience'}),
        ('--feature-scale', {'nargs': '?', 'type': int,
         'default': 1, 'help': 'Divider for # of features to use'}),
        ('--weights', {'nargs': '?', 'type': str, 'default': None,
         'help': 'Path to previously trained model weights file .pkl'}),
        ('--furukawa-weights', {'nargs': '?', 'type': str, 'default': None,
         'help': 'Path to previously trained furukawa model weights file .pkl'}),
        ('--new-hyperparams', {'nargs': '?', 'type': bool, 'default': False,
         'const': True, 'help': 'Continue training with new hyperparameters'}),
        ('--log-path', {'nargs': '?', 'type': str,
         'default': 'runs_cubi/', 'help': 'Path to log directory'}),
        ('--debug', {'nargs': '?', 'type': bool,
         'default': False, 'const': True, 'help': 'Debug mode'}),
        ('--plot-samples', {'nargs': '?', 'type': bool, 'default': False,
         'const': True, 'help': 'Plot floorplan segmentations to Tensorboard.'}),
        ('--scale', {'nargs': '?', 'type': bool, 'default': False,
         'const': True, 'help': 'Rescale to 256x256 augmentation.'})
    ]
    parser = argparse.ArgumentParser(description='Hyperparameters')
    [parser.add_argument(arg, **specs) for arg, specs in arg_specs]

    return parser.parse_args()


def setup_logging(args):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_dir = os.path.join(args.log_path, time_stamp)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    logger = setup_logger(log_dir)
    return log_dir, writer, logger


def setup_logger(log_dir):
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    # Setup file handler and formatter...
    return logger


if __name__ == '__main__':
    args = parse_arguments()
    log_dir, writer, logger = setup_logging(args)
    train(args, log_dir, writer, logger)
