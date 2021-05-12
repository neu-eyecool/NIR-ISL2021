import numpy as np
from hausdorff import hausdorff_distance

  
def compute_tfpn(pred_mask,true_mask):

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


def compute_e1(n_batch, pred_masks, true_masks):

    sum_e1 = 0
    for i in range(n_batch):
        tpfn = compute_tfpn(pred_masks[i],true_masks[i])
        fp, fn = tpfn['fp'], tpfn['fn']
        sum_e1 += fp+fn

    return sum_e1/n_batch


def compute_miou(n_batch, true_masks, pred_masks):

    sum_iou = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if tp+fn+fp == 0:
            iou=1
        else:
            iou=tp/(tp+fn+fp)
        sum_iou += iou

    return sum_iou/n_batch


def compute_dice(n_batch, pred_masks, true_masks):

    sum_dice = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if 2*tp+fn+fp == 0:
            dice=1
        else:
            dice=2*tp/(2*tp+fn+fp)
        sum_dice += dice

    return sum_dice/n_batch


def compute_f1(n_batch, pred_masks, true_masks):

    sum_f1 = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if tp+fp == 0:
            precision = tp
        else:
            precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        if precision+recall == 0:
            f1 = tp
        else:
            f1 = (2*precision*recall) / (precision+recall)
        if f1 > 999:
            f1 = 0
        sum_f1 += f1

    return sum_f1/n_batch


def get_coords(nparray):
    coords = []

    h, w = nparray.shape
    for i in range(h):
        for j in range(w):
            if nparray[i, j] > 0:
                coords.append([i, j])
    
    return np.asarray(coords)


def Hausdorff(pred_edge, true_edge):
    pred_edge = np.asarray(pred_edge>0)
    true_edge = np.asarray(true_edge>0)
    _, h, w = true_edge.shape

    pred_coords = get_coords(pred_edge[0])
    true_coords = get_coords(true_edge[0])

    if len(pred_coords) == 0 or len(true_coords)==0:
        hsdf =  float("inf")
    else:
        hsdf = hausdorff_distance(pred_coords, true_coords) / w

    return hsdf


def compute_hsdf(n_batch, pred_edges, true_edges):
    hsdf = 0
    for i in range(n_batch):
        hsdf_i = Hausdorff(pred_edges[i], true_edges[i])
        if hsdf_i == float("inf"):
            continue
        hsdf += hsdf_i
    return hsdf/n_batch


def evaluate_loc(pred_masks, true_masks, pred_edges, true_edges, dataset_name):

    n_batch = true_masks.size()[0]
    
    e1 = compute_e1(n_batch, pred_masks, true_masks).item()
    dice = compute_dice(n_batch, pred_masks, true_masks).item()
    iou = compute_miou(n_batch, pred_masks, true_masks).item()

    # # caculate hausdorff takes too long
    # hsdf = compute_hsdf(n_batch, pred_edges.cpu(), true_edges.cpu())

    return {
        'E1': e1*100,
        'IoU': iou*100,
        'Dice': dice,
        # 'Hsdf': hsdf*100
        
    }
