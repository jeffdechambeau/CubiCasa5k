from floortrans.metrics import get_px_acc
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import torch
import logging


def validate_epoch(epoch, model, valloader,  criterion, input_slice, running_metrics_room_val, running_metrics_icon_val, loss, writer):
    model.eval()
    val_losses = pd.DataFrame()
    val_variances = pd.DataFrame()
    val_ss = pd.DataFrame()
    px_rooms = 0
    px_icons = 0
    total_px = 0

    for i_val, samples_val in tqdm(enumerate(valloader), total=len(valloader), ncols=80, leave=False):
        with torch.no_grad():
            images_val = samples_val['image'].cuda(non_blocking=True)
            labels_val = samples_val['label'].cuda(non_blocking=True)

            outputs = model(images_val)
            labels_val = F.interpolate(
                labels_val, size=outputs.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, labels_val)

            start, end = input_slice

            room_pred = outputs[0, start:start +
                                end].argmax(0).data.cpu().numpy()
            room_gt = labels_val[0, start].data.cpu().numpy()
            running_metrics_room_val.update(room_gt, room_pred)

            icon_pred = outputs[0, start +
                                end:].argmax(0).data.cpu().numpy()
            icon_gt = labels_val[0, start+1].data.cpu().numpy()
            running_metrics_icon_val.update(icon_gt, icon_pred)
            total_px += outputs[0, 0].numel()
            pr, pi = get_px_acc(outputs[0], labels_val[0], input_slice, 0)
            px_rooms += float(pr)
            px_icons += float(pi)

            val_losses = val_losses.append(
                criterion.get_loss(), ignore_index=True)
            val_variances = val_variances.append(
                criterion.get_var(), ignore_index=True)
            val_ss = val_ss.append(criterion.get_s(), ignore_index=True)

    val_loss = val_losses.mean()
    val_variance = val_variances.mean()

    logging.info("val_loss: "+str(val_loss))
    writer.add_scalars('validation loss', val_loss, global_step=1+epoch)
    writer.add_scalars('validation variance',
                       val_variance, global_step=1+epoch)
    return val_loss, val_variance, val_variances, val_ss
