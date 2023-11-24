def add_scalars(writer, epoch, room_score, room_class_iou, running_metrics_room_val, running_metrics_icon_val, val_loss, val_variance, val_s):
    writer.add_scalars('validation/room/general',
                       room_score, global_step=1+epoch)
    writer.add_scalars('validation/room/IoU',
                       room_class_iou['Class IoU'], global_step=1+epoch)
    writer.add_scalars('validation/room/Acc',
                       room_class_iou['Class Acc'], global_step=1+epoch)
    running_metrics_room_val.reset()

    icon_score, icon_class_iou = running_metrics_icon_val.get_scores()
    writer.add_scalars('validation/icon/general',
                       icon_score, global_step=1+epoch)
    writer.add_scalars('validation/icon/IoU',
                       icon_class_iou['Class IoU'], global_step=1+epoch)
    writer.add_scalars('validation/icon/Acc',
                       icon_class_iou['Class Acc'], global_step=1+epoch)
    running_metrics_icon_val.reset()

    writer.add_scalars('validation/loss', val_loss, global_step=1+epoch)
    writer.add_scalars('validation/variance',
                       val_variance, global_step=1+epoch)
    writer.add_scalars('validation/s', val_s, global_step=1+epoch)
