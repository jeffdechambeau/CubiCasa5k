import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import logging
import argparse
from tqdm import tqdm

# Import your specific functions from other modules here
from floortrans.models import get_model
from floortrans.losses import UncertaintyLoss
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import Compose, DictToTensor, ColorJitterTorch, RandomRotations, RandomChoice, RandomCropToSizeTorch, ResizePaddedTorch
from floortrans.metrics import runningScore, get_px_acc

# Helper function to set up the data augmentation pipeline
def setup_augmentations(args):
    if args.scale:
        return Compose([RandomChoice([RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                                      ResizePaddedTorch((0, 0), data_format='dict', size=(args.image_size, args.image_size))]),
                        RandomRotations(format='cubi'),
                        DictToTensor(),
                        ColorJitterTorch()])
    else:
        return Compose([RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                        RandomRotations(format='cubi'),
                        DictToTensor(),
                        ColorJitterTorch()])

# Helper function for model setup
def setup_model(args, logger):
    model = get_model(args.arch, args.n_classes)
    criterion = UncertaintyLoss(input_slice=[21, 12, 11])

    if args.arch == 'hg_furukawa_original' and args.furukawa_weights:
        logger.info(f"Loading Furukawa model weights from checkpoint '{args.furukawa_weights}'")
        checkpoint = torch.load(args.furukawa_weights)
        model.load_state_dict(checkpoint['model_state'])
        criterion.load_state_dict(checkpoint['criterion_state'])

    model.conv4_ = torch.nn.Conv2d(256, args.n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(args.n_classes, args.n_classes, kernel_size=4, stride=4)
    nn.init.kaiming_normal_(model.conv4_.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(model.conv4_.bias, 0)
    nn.init.kaiming_normal_(model.upsample.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(model.upsample.bias, 0)

    return model, criterion

# Helper function to configure optimizer
def configure_optimizer(args, model, criterion):
    params = [{'params': model.parameters(), 'lr': args.l_rate},
              {'params': criterion.parameters(), 'lr': args.l_rate_var}]

    if args.optimizer.startswith('adam'):
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
        scheduler = None
        if 'patience' in args.optimizer:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=0.5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=10**-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch/args.n_epoch)**0.9)
    
    return optimizer, scheduler

# Main train function
def train(args):
    log_dir = f"{args.log_path}/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)
    logger = setup_logger(log_dir)

    with open(f'{log_dir}/args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    aug = setup_augmentations(args)
    train_set, val_set = setup_datasets(args, aug)
    trainloader, valloader = setup_dataloaders(args, train_set, val_set)
    
    model, criterion = setup_model(args, logger)
    optimizer, scheduler = configure_optimizer(args, model, criterion)

    model = model.cuda()
    criterion = criterion.cuda()
    log_model_graph(writer, model, args)

    best_metrics = initialize_best_metrics()
    start_epoch, no_improvement = load_weights(args, model, criterion, optimizer, logger)

    for epoch in range(start_epoch, args.n_epoch):
        train_epoch(model, criterion, optimizer, trainloader, writer, logger, epoch, args)
        val_metrics = validate(model, criterion, valloader, writer, logger, epoch, args)

        update_best_metrics(best_metrics, val_metrics, model, criterion, optimizer, log_dir, writer, logger, epoch, args)
        adjust_learning_rate(args, scheduler, val_metrics, best_metrics, model, optimizer, logger, no_improvement)

    logger.info("Training complete.")
    save_final_model(model, criterion, optimizer, log_dir, epoch)

def setup_logger(log_dir):
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(f'{log_dir}/train.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

def setup_datasets(args, aug):
    train_set = FloorplanSVG(args.data_path, 'train.txt', format='lmdb', augmentations=aug)
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb', augmentations=DictToTensor())
    return train_set, val_set

def setup_dataloaders(args, train_set, val_set):
    num_workers = 0 if args.debug else 8
    trainloader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    valloader = data.DataLoader(val_set, batch_size=1, num_workers=num_workers, pin_memory=True)
    return trainloader, valloader

def log_model_graph(writer, model, args):
    dummy_input = torch.zeros((2, 3, args.image_size, args.image_size)).cuda()
    writer.add_graph(model, dummy_input)

def initialize_best_metrics():
    return {
        'best_loss': np.inf,
        'best_loss_var': np.inf,
        'best_train_loss': np.inf,
        'best_acc': 0,
        'best_val_loss_variance': np.inf,
        'no_improvement': 0
    }

def load_weights(args, model, criterion, optimizer, logger):
    start_epoch = 0
    if args.weights and os.path.exists(args.weights):
        logger.info(f"Loading model and optimizer from checkpoint '{args.weights}'")
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model_state'])
        criterion.load_state_dict(checkpoint['criterion_state'])
        if not args.new_hyperparams:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            logger.info("Using old optimizer state.")
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Loaded checkpoint '{args.weights}' (epoch {start_epoch})")
    else:
        logger.info(f"No checkpoint found at '{args.weights}'")
    return start_epoch, 0

def train_epoch(model, criterion, optimizer, trainloader, writer, logger, epoch, args):
    model.train()
    losses, variances, ss = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i, samples in tqdm(enumerate(trainloader), total=len(trainloader), ncols=80, leave=False):
        images = samples['image'].cuda(non_blocking=True)
        labels = samples['label'].cuda(non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses = pd.concat([losses, criterion.get_loss()], ignore_index=True)
        variances = pd.concat([variances, criterion.get_var()], ignore_index=True)
        ss = pd.concat([ss, criterion.get_s()], ignore_index=True)

    log_training_metrics(writer, losses, variances, ss, epoch)

def validate(model, criterion, valloader, writer, logger, epoch, args):
    model.eval()
    val_losses, val_variances, val_ss = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    running_metrics_room_val = runningScore(12)
    running_metrics_icon_val = runningScore(11)


    for i_val, samples_val in tqdm(enumerate(valloader), total=len(valloader), ncols=80, leave=False):
        with torch.no_grad():
            # ... [previous code] ...
            update_validation_metrics(val_losses, val_variances, val_ss, criterion, running_metrics_room_val, running_metrics_icon_val, outputs, labels_val)

    val_loss = val_losses.mean()
    val_variance = val_variances.mean()
    val_s = val_ss.mean()
    room_score, room_class_iou = running_metrics_room_val.get_scores()
    icon_score, icon_class_iou = running_metrics_icon_val.get_scores()
    running_metrics_room_val.reset()
    running_metrics_icon_val.reset()

    log_validation_metrics(writer, val_loss, val_variance, val_s, room_score, room_class_iou, icon_score, icon_class_iou, epoch)

    return {
        'val_loss': val_loss,
        'val_variance': val_variance,
        'room_score': room_score,
        'icon_score': icon_score
    }

def update_validation_metrics(val_losses, val_variances, val_ss, criterion, running_metrics_room_val, running_metrics_icon_val, outputs, labels_val):
    room_pred = outputs[0, 21:33].argmax(0).data.cpu().numpy()
    room_gt = labels_val[0, 21].data.cpu().numpy()
    running_metrics_room_val.update(room_gt, room_pred)

    icon_pred = outputs[0, 33:].argmax(0).data.cpu().numpy()
    icon_gt = labels_val[0, 22].data.cpu().numpy()
    running_metrics_icon_val.update(icon_gt, icon_pred)

    val_losses = pd.concat([val_losses, criterion.get_loss()], ignore_index=True)
    val_variances = pd.concat([val_variances, criterion.get_var()], ignore_index=True)
    val_ss = pd.concat([val_ss, criterion.get_s()], ignore_index=True)

def log_validation_metrics(writer, val_loss, val_variance, val_s, room_score, room_class_iou, icon_score, icon_class_iou, epoch):
    writer.add_scalars('validation/loss', val_loss, global_step=epoch+1)
    writer.add_scalars('validation/variance', val_variance, global_step=epoch+1)
    writer.add_scalars('validation/s', val_s, global_step=epoch+1)
    writer.add_scalars('validation/room/general', room_score, global_step=epoch+1)
    writer.add_scalars('validation/room/IoU', room_class_iou['Class IoU'], global_step=epoch+1)
    writer.add_scalars('validation/room/Acc', room_class_iou['Class Acc'], global_step=epoch+1)
    writer.add_scalars('validation/icon/general', icon_score, global_step=epoch+1)
    writer.add_scalars('validation/icon/IoU', icon_class_iou['Class IoU'], global_step=epoch+1)
    writer.add_scalars('validation/icon/Acc', icon_class_iou['Class Acc'], global_step=epoch+1)

def adjust_learning_rate(args, scheduler, val_metrics, best_metrics, model, optimizer, logger, no_improvement):
    # Code for adjusting learning rate based on validation metrics and updating best_metrics

def save_final_model(model, criterion, optimizer, log_dir, epoch):
    state = {'epoch': epoch + 1, 'model_state': model.state_dict(), 'criterion_state': criterion.state_dict(), 'optimizer_state': optimizer.state_dict()}
    torch.save(state, os.path.join(log_dir, "model_last_epoch.pkl"))

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    train(args)

import argparse

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Hyperparameters for Floorplan Segmentation Model Training')
    
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use [\'hg_furukawa_original, segnet, etc.\']')
    parser.add_argument('--optimizer', nargs='?', type=str, default='adam-patience-previous-best',
                        help='Optimizer to use [\'adam, sgd\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='Number of classes for the model')
    parser.add_argument('--n-epoch', nargs='?', type=int, default=1000,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', nargs='?', type=int, default=26,
                        help='Batch Size for training')
    parser.add_argument('--image-size', nargs='?', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--l-rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--l-rate-var', nargs='?', type=float, default=1e-3,
                        help='Learning Rate for Variance')
    parser.add_argument('--l-rate-drop', nargs='?', type=float, default=200,
                        help='Epochs after which learning rate drops')
    parser.add_argument('--patience', nargs='?', type=int, default=10,
                        help='Patience for learning rate reduction')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file (.pkl)')
    parser.add_argument('--furukawa-weights', nargs='?', type=str, default=None,
                        help='Path to previously trained furukawa model weights file (.pkl)')
    parser.add_argument('--new-hyperparams', action='store_true',
                        help='Continue training with new hyperparameters')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with verbose logging')
    parser.add_argument('--plot-samples', action='store_true',
                        help='Plot floorplan segmentations to Tensorboard')
    parser.add_argument('--scale', action='store_true',
                        help='Rescale to 256x256 as augmentation')
    
    return parser

if __name__ == '__main__':
    main()
