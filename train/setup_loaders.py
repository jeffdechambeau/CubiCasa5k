
from floortrans.loaders.augmentations import (RandomCropToSizeTorch,
                                              ResizePaddedTorch,
                                              Compose,
                                              DictToTensor,
                                              ColorJitterTorch,
                                              RandomRotations)

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import RandomChoice
import torch
from torch.utils import data
from floortrans.loaders import FloorplanSVG
import logging


def setup_optimizer(args, model, criterion):
    params = [{'params': model.parameters(), 'lr': args.l_rate},
              {'params': criterion.parameters(), 'lr': args.l_rate}]

    if args.optimizer == 'adam-patience':
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', patience=args.patience, factor=0.5)
    elif args.optimizer == 'adam-patience-previous-best':
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
    elif args.optimizer == 'sgd':
        def lr_drop(epoch):
            return (1 - epoch/args.n_epoch)**0.9
        optimizer = torch.optim.SGD(
            params, momentum=0.9, weight_decay=10**-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_drop)
    elif args.optimizer == 'adam-scheduler':
        def lr_drop(epoch):
            return 0.5 ** np.floor(epoch / args.l_rate_drop)
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_drop)

    return optimizer, scheduler


def setup_augmentations(args):
    if args.scale:
        return Compose([RandomChoice([RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                                      ResizePaddedTorch((0, 0), data_format='dict', size=(args.image_size, args.image_size))]),
                        RandomRotations(format='cubi'),
                        DictToTensor(),
                        ColorJitterTorch()])

    return Compose([RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                    RandomRotations(format='cubi'),
                    DictToTensor(),
                    ColorJitterTorch()])


def setup_dataloader(args, aug, writer):

    writer.add_text('parameters', str(vars(args)))
    logging.info('Loading data...')
    train_set = FloorplanSVG(args.data_path, 'train.txt', format='lmdb',
                             augmentations=aug)
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb',
                           augmentations=DictToTensor())

    num_workers = 0 if args.debug else 8

    trainloader = data.DataLoader(train_set, batch_size=args.batch_size,
                                  num_workers=num_workers, shuffle=True, pin_memory=True)
    valloader = data.DataLoader(val_set, batch_size=1,
                                num_workers=num_workers, pin_memory=True)
    return trainloader, valloader
