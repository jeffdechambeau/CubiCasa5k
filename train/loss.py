
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import numpy as np


def plot_samples(args, valloader, writer, model, input_slice, epoch, first_best):

    if not args.plot_samples:
        return

    for i, samples_val in enumerate(valloader):
        with torch.no_grad():
            if i == 4:
                break

            images_val = samples_val['image'].cuda(
                non_blocking=True)
            labels_val = samples_val['label'].cuda(
                non_blocking=True)

            if first_best:
                writer.add_image("Image "+str(i), images_val[0])
                for j, l in enumerate(labels_val.squeeze().cpu().data.numpy()):
                    fig = plt.figure(figsize=(18, 12))
                    plot = fig.add_subplot(111)
                    if j < 21:
                        cax = plot.imshow(l, vmin=0, vmax=1)
                    else:
                        cax = plot.imshow(
                            l, vmin=0, vmax=19, cmap=plt.cm.tab20)
                    fig.colorbar(cax)
                    writer.add_figure(
                        "Image "+str(i)+" label/Channel "+str(j), fig)

            outputs = model(images_val)

            pred_arr = torch.split(outputs, input_slice, 1)
            heatmap_pred, rooms_pred, icons_pred = pred_arr

            rooms_pred = softmax(rooms_pred, 1).cpu().data.numpy()
            icons_pred = softmax(icons_pred, 1).cpu().data.numpy()

            label = "Image "+str(i)+" prediction/Channel "

            for j, l in enumerate(np.squeeze(heatmap_pred)):
                fig = plt.figure(figsize=(18, 12))
                plot = fig.add_subplot(111)
                cax = plot.imshow(l, vmin=0, vmax=1)
                fig.colorbar(cax)
                writer.add_figure(
                    label+str(j), fig, global_step=1+epoch)

            fig = plt.figure(figsize=(18, 12))
            plot = fig.add_subplot(111)
            cax = plot.imshow(np.argmax(np.squeeze(
                rooms_pred), axis=0), vmin=0, vmax=19, cmap=plt.cm.tab20)
            fig.colorbar(cax)
            writer.add_figure(label+str(j+1), fig,
                              global_step=1+epoch)

            fig = plt.figure(figsize=(18, 12))
            plot = fig.add_subplot(111)
            cax = plot.imshow(np.argmax(np.squeeze(
                icons_pred), axis=0), vmin=0, vmax=19, cmap=plt.cm.tab20)
            fig.colorbar(cax)
            writer.add_figure(label+str(j+2), fig,
                              global_step=1+epoch)


def save_loss(args, avg_loss, val_loss, logger, writer, model, epoch, criterion, optimizer, log_dir, valloader, input_slice, room_score, icon_score):
    if val_loss['total loss with variance'] < best_loss_var:
        best_loss_var = val_loss['total loss with variance']
        logger.info(
            "Best validation loss with variance found saving model...")
        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),
                 'criterion_state': criterion.state_dict(),
                 'optimizer_state': optimizer.state_dict(),
                 'best_loss': best_loss}
        torch.save(state, log_dir+"/model_best_val_loss_var.pkl")

        plot_samples(args, valloader, writer, model, input_slice)

    if val_loss['total loss'] < best_loss:
        best_loss = val_loss['total loss']
        logger.info("Best validation loss found saving model...")
        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),
                 'criterion_state': criterion.state_dict(),
                 'optimizer_state': optimizer.state_dict(),
                 'best_loss': best_loss}
        torch.save(state, log_dir+"/model_best_val_loss.pkl")

    px_acc = room_score["Mean Acc"] + icon_score["Mean Acc"]

    if px_acc > best_acc:
        best_acc = px_acc
        logger.info("Best validation pixel accuracy found saving model...")
        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),
                 'criterion_state': criterion.state_dict(),
                 'optimizer_state': optimizer.state_dict()}
        torch.save(state, log_dir+"/model_best_val_acc.pkl")

    if avg_loss < best_train_loss:
        best_train_loss = avg_loss
        logger.info("Best training loss with variance...")
        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),
                 'criterion_state': criterion.state_dict(),
                 'optimizer_state': optimizer.state_dict()}
        torch.save(state, log_dir+"/model_best_train_loss_var.pkl")

    logger.info("Last epoch done saving final model...")

    state = {'epoch': epoch+1,
             'model_state': model.state_dict(),
             'criterion_state': criterion.state_dict(),
             'optimizer_state': optimizer.state_dict()}

    torch.save(state, log_dir+"/model_last_epoch.pkl")
