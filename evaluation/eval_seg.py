

def compute_tfpn(true_mask, pred_mask):

    c, r = true_mask.shape[1], pred_mask.shape[2]
    num_pixel = c*r

    true_mask = true_mask>0
    pred_mask = pred_mask>0
    
    tp = (true_mask & pred_mask).sum()
    fp = (~true_mask & pred_mask).sum()
    tn = (~(true_mask | pred_mask)).sum()
    fn = (true_mask & (~pred_mask)).sum()

    return {
        'tp': tp/num_pixel,
        'fp': fp/num_pixel,
        'tn': tn/num_pixel,        
        'fn': fn/num_pixel
    }


def compute_e1(n_batch, true_masks, pred_masks):

    sum_e1 = 0
    for i in range(n_batch):
        tpfn = compute_tfpn(true_masks[i], pred_masks[i])
        fp, fn = tpfn['fp'], tpfn['fn']
        sum_e1 += fp+fn

    return sum_e1/n_batch


def compute_miou(n_batch, true_masks, pred_masks):

    sum_iou = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(true_masks[i], pred_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if tp+fn+fp == 0:
            iou=1
        else:
            iou=tp/(tp+fn+fp)
        sum_iou += iou

    return sum_iou/n_batch

def compute_dice(n_batch, true_masks, pred_masks):

    sum_dice = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(true_masks[i], pred_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if 2*tp+fn+fp == 0:
            dice=1
        else:
            dice=2*tp/(2*tp+fn+fp)
        sum_dice += dice

    return sum_dice/n_batch


def evaluate_seg(pred_masks, true_masks, dataset_name):

    n_batch = true_masks.size()[0]
    e1, iou, dice = 0, 0, 0

    # compute miou, dice, recall, e1, f1
    iou = compute_miou(n_batch, true_masks, pred_masks).item()
    dice = compute_dice(n_batch, true_masks, pred_masks).item()
    e1 = compute_e1(n_batch, true_masks, pred_masks).item()

    return {
        'E1': e1*100,
        'IoU': iou*100,
        'Dice': dice
    }
