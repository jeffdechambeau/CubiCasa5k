from tqdm import tqdm
import numpy as np
import logging


def train_epoch(model, criterion, epoch, trainloader, losses, lossess, optimizer, variances, ss, writer, args):
    for i, samples in tqdm(enumerate(trainloader), total=len(trainloader), ncols=80, leave=False):
        images = samples['image'].cuda(non_blocking=True)
        labels = samples['label'].cuda(non_blocking=True)

        outputs = model(images)

        loss = criterion(outputs, labels)
        lossess.append(loss.item())
        losses = losses.append(criterion.get_loss(), ignore_index=True)
        variances = variances.append(
            criterion.get_var(), ignore_index=True)
        ss = ss.append(criterion.get_s(), ignore_index=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(lossess)
    avg_loss = np.inf
    loss = losses.mean()
    variance = variances.mean()
    s = ss.mean()

    logging.info("Epoch [%d/%d] Loss: %.4f" %
                 (epoch+1, args.n_epoch, avg_loss))

    writer.add_scalars('training/loss', loss, global_step=1+epoch)
    writer.add_scalars('training/variance', variance, global_step=1+epoch)
    writer.add_scalars('training/s', s, global_step=1+epoch)
    current_lr = {'base': optimizer.param_groups[0]['lr'],
                  'var': optimizer.param_groups[1]['lr']}
    writer.add_scalars('training/lr', current_lr, global_step=1+epoch)

    return avg_loss, loss, variance
