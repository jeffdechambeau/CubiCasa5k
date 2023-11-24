

import logging
import torch.nn as nn
import torch
from floortrans.losses import UncertaintyLoss
from floortrans.models import get_model


def load_model(args, input_slice, logger):
    logging.info('Loading model...')

    if args.arch is not 'hg_furukawa_original':
        model = get_model(args.arch, args.n_classes)
        criterion = UncertaintyLoss(input_slice=input_slice)
        return model, criterion

    model = get_model(args.arch, 51)
    criterion = UncertaintyLoss(input_slice=input_slice)

    if args.furukawa_weights:
        logger.info("Loading furukawa model weights from checkpoint '{}'".format(
            args.furukawa_weights))
        checkpoint = torch.load(args.furukawa_weights)
        model.load_state_dict(checkpoint['model_state'])
        criterion.load_state_dict(checkpoint['criterion_state'])

    model.conv4_ = torch.nn.Conv2d(
        256, args.n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(
        args.n_classes, args.n_classes, kernel_size=4, stride=4)

    for m in [model.conv4_, model.upsample]:
        nn.init.kaiming_normal_(
            m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)

    return model, criterion
