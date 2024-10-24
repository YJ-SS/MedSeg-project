import torch

def compute_dice_coffeient(seg1, seg2)->float:
    '''

    :param seg1:
    :param seg2:
    :return:
    '''
    return 2 * (seg1 & seg2).sum() / (seg1.sum() + seg2.sum())

def get_dice(y_pred, y_true, num_clus)->tuple[float,list[float]]:
    '''
    Calculate dice between predicted and ground truth
    :param y_pred: Shape as [B,D,H,W] default channel is 1, ignored
    :param y_true: Shape as [B,C,D,H,W], which C is num_clus
    :param num_clus: Equal to C
    :return: Average Dice of all class and dice of each class
    '''
    dice_matrix = [0 for i in range(num_clus)]
    avg_dice = 0
    cnt = 0
    # y_pred shape: [B,C,D,H,W] -> [B,D,H,W]
    y_pred = torch.argmax(y_pred, dim=1).squeeze(dim=1).detach().cpu().numpy()
    y_true = y_true.squeeze(dim=1).detach().cpu().numpy()
    for i in range(0, num_clus):
        if (y_true == i).sum() == 0 or (y_true == i).sum() == 0:
            continue
        else:
            dice_matrix[i] = compute_dice_coffeient((y_pred == i), (y_true == i))
            if i != 0:
                # Do not contain label 0
                avg_dice += dice_matrix[i]
                cnt += 1
    avg_dice /= cnt
    return avg_dice, dice_matrix
