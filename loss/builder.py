import torch
import torch.nn as nn

from .ohem_loss import OhemCELoss
from .dice_loss import SoftDiceLoss
from .focal_loss import FocalLoss


class Make_Criterion(nn.Module):
    def __init__(self, deep_supervise):
        super(Make_Criterion, self).__init__()
        self.aux_num = deep_supervise
        # self.bceloss = nn.BCEWithLogitsLoss().cuda()
        # self.focalloss = FocalLoss().cuda()
        self.diceloss = SoftDiceLoss().cuda()

    def forward(self, pred, label):
        loss_bce, loss_dice, loss_fl = 0, 0, 0 
        
        if self.aux_num == 1:
            B = pred.size()[0]
            # loss_bce = self.bceloss(pred, label) / B
            loss_dice = self.diceloss(pred, label) / B
            # loss_fl = self.focalloss(pred, label) / B
        else:
            B = pred[0].size()[0]
            # loss_bce = sum([self.bceloss(pred_i, label) for pred_i in pred]) / B
            loss_dice = sum([self.diceloss(pred_i, label) for pred_i in pred]) / B
            # loss_fl = sum([self.focalloss(pred_i, label) for pred_i in pred]) / B

        loss = loss_dice 
        return loss