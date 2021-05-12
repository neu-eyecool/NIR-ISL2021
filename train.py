import os
import logging
from datetime import datetime
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A

import sys
sys.path.append('.../NIR-ISL2021master/')
from datasets import nirislDataset
from models import EfficientUNet
from loss import Make_Criterion
from evaluation import evaluate_loc, evaluate_seg
from location import get_edge


os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
cudnn.benchmark = True

experiment_name = 'M1-e5UNet'
dataset_name = 'CASIA-Iris-Mobile-V1.0'
assert dataset_name in ['CASIA-Iris-Africa','CASIA-distance', 'Occlusion', 'Off_angle', 'CASIA-Iris-Mobile-V1.0']


def get_args():
    parser = argparse.ArgumentParser(description='Train paprmeters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', type=int, default=96, dest='epoch_num')
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=8, dest='batch_size')
    parser.add_argument('-l', '--learning-rate', type=float, nargs='?', default=0.002, dest='lr')
    parser.add_argument('--log', type=str, default='logging.log', dest='log_name')
    parser.add_argument('--ckp', type=str, default=None, help='load a pertrain model from .../xxx.pth', dest='checkpoints')
    parser.add_argument('--gpu-ids', type=int, nargs='*', help='use cuda', dest='gpu_ids')
    return parser.parse_args()


def main(train_args):
    ########################################### logging and writer #############################################
    writer = SummaryWriter(log_dir=os.path.join(log_path, 'summarywriter_'+train_args['log_name'].split('.')[0]), comment=train_args['log_name'])

    logging.info('------------------------------------------------train configs------------------------------------------------')
    logging.info(train_args)

    ############################################# define a CNN #################################################
    net = EfficientUNet(num_classes=3).cuda()
    if train_args['checkpoints']:
        net.load_state_dict(torch.load(train_args['checkpoints']))
    if train_args['gpu_ids']:
        net = torch.nn.DataParallel(net, device_ids=train_args['gpu_ids'])

    ########################################### dataset #############################################
    train_augment = A.Compose([
        # A.Resize(320, 544), # for Africa dataset
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.ShiftScaleRotate(p=0.4),
        A.Equalize(p=0.2)
    ])
    val_augment = A.Compose([
        # A.Resize(320, 544) # for Africa dataset
    ])
    train_dataset = nirislDataset(dataset_name, mode='train', transform=train_augment)
    val_dataset = nirislDataset(dataset_name, mode='val', transform=val_augment)
    train_loader = DataLoader(train_dataset, batch_size=train_args['batch_size'], num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_args['batch_size'], num_workers=8, drop_last=True)

    logging.info(f'data_augment are: \n {train_augment} \n {val_augment}')
    logging.info(f'The dataset {dataset_name} is ready!')

    ########################################### criterion #############################################
    criterion = Make_Criterion(deep_supervise = 1)
    heatmap_criteria = torch.nn.MSELoss().cuda()
    logging.info(f'''criterion is ready! \n{criterion} \n{heatmap_criteria}''')

    ########################################### optimizer #############################################
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': 1e-8}
    ], betas=(0.95, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8, min_lr=1e-10, verbose=True)
    logging.info(f'optimizer is ready! \n{optimizer}')

    ######################################### train and val ############################################
    net.train()
    try:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 999, 'E1': 999, 'IoU': 0, 'Dice': 0, 'F1': 0}
        train_args['best_record_inner'] = {'epoch': 0, 'val_loss': 999, 'E1': 999, 'IoU': 0, 'Dice': 0}
        train_args['best_record_outer'] = {'epoch': 0, 'val_loss': 999, 'E1': 999, 'IoU': 0, 'Dice': 0}

        for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
            train(writer, train_loader, net, criterion, heatmap_criteria, optimizer, epoch, train_args)
            val_loss = validate(writer, val_loader, net, criterion, optimizer, epoch, train_args)
            scheduler.step(val_loss)

        writer.close()
        logging.info('-------------------------------------------------best record------------------------------------------------')
        logging.info('mask    epoch:{}   val loss {:.5f}  E1:{:.5f}   IoU:{:.5f}   Dice:{:.5f}  F1:{:.5f}'.format(
            train_args['best_record']['epoch'], train_args['best_record']['val_loss'], train_args['best_record']['E1'],
            train_args['best_record']['IoU'], train_args['best_record']['Dice'], train_args['best_record']['F1']
            ))
        logging.info('outer   epoch:{}  val loss {:.5f}  E1:{:.5f}   IoU:{:.5f}   Dice:{:.5f}'.format(
            train_args['best_record_outer']['epoch'], train_args['best_record_outer']['val_loss'], train_args['best_record_outer']['E1'],
            train_args['best_record_outer']['IoU'], train_args['best_record_outer']['Dice']
            ))
        logging.info('inner   epoch:{}  val loss {:.5f}  E1:{:.5f}   IoU:{:.5f}   Dice:{:.5f}'.format(
            train_args['best_record_inner']['epoch'], train_args['best_record_inner']['val_loss'], train_args['best_record_inner']['E1'],
            train_args['best_record_inner']['IoU'], train_args['best_record_inner']['Dice']
            ))

    except KeyboardInterrupt:
        torch.save(net.module.state_dict(), log_path+'/INTERRUPTED.pth')
        logging.info('Saved interrupt!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def train(writer, train_loader, net, criterion, heatmap_criteria, optimizer, epoch, train_args):
    logging.info('--------------------------------------------------training...------------------------------------------------')
    iters = len(train_loader)
    curr_iter = (epoch - 1) * iters

    for i, data in enumerate(train_loader):
        image, mask, iris_mask, pupil_mask, heatmap = \
            data['image'], data['mask'], data['iris_edge_mask'], data['pupil_edge_mask'], data['heatmap'] #BCHW
        
        assert image.size()[2:] == mask.size()[2:]
        image = Variable(image).cuda()
        mask = Variable(mask).cuda()
        iris_mask = Variable(iris_mask).cuda()
        pupil_mask = Variable(pupil_mask).cuda()
        heatmap = Variable(heatmap).cuda()

        optimizer.zero_grad()
        outputs = net(image)
        pred_mask, pred_iris_mask, pred_pupil_mask, pred_heatmap = \
            outputs['pred_mask'], outputs['pred_iris_mask'], outputs['pred_pupil_mask'], outputs['heatmap']

        loss_mask = criterion(pred_mask, mask)
        loss_iris = criterion(pred_iris_mask, iris_mask)
        loss_pupil = criterion(pred_pupil_mask, pupil_mask)

        heatmap0 = transforms.Resize((pred_heatmap[0].size()[2:]))(heatmap)
        heatmap1 = transforms.Resize((pred_heatmap[1].size()[2:]))(heatmap)
        heatmap2 = transforms.Resize((pred_heatmap[2].size()[2:]))(heatmap)
        loss_heatmap = heatmap_criteria(pred_heatmap[0], heatmap0) + heatmap_criteria(pred_heatmap[1], heatmap1) + heatmap_criteria(pred_heatmap[2], heatmap2)

        loss = loss_mask + loss_iris + 2*loss_pupil + 0.3*loss_heatmap

        loss.backward()
        optimizer.step()

        writer.add_scalar('train_loss/iter', loss.item(), curr_iter)
        writer.add_scalar('train_loss_mask/iter', loss_mask.item(), curr_iter)
        writer.add_scalar('train_loss_iris/iter', loss_iris.item(), curr_iter)
        writer.add_scalar('train_loss_pupil/iter', loss_pupil.item(), curr_iter)
        writer.add_scalar('train_loss_heatmap/iter', loss_heatmap.item(), curr_iter)

        if (i + 1) % train_args['print_freq'] == 0:
            print('epoch:{:2d}  iter/iters:{:3d}/{:3d}  train_loss:{:.9f}  loss_mask:{:.9f}  loss_iris:{:.9}   loss_pupil:{:.9}    loss_heatmap:{:.9f}'.format(
                epoch, i+1, iters, loss, loss_mask, loss_iris, loss_pupil, loss_heatmap))
            logging.info('epoch:{:2d}  iter/iters:{:3d}/{:3d}  train_loss:{:.9f}  loss_mask:{:.9f}  loss_iris:{:.9}   loss_pupil:{:.9}    loss_heatmap:{:.9f}'.format(
                epoch, i+1, iters, loss, loss_mask, loss_iris, loss_pupil, loss_heatmap))

        curr_iter += 1


def validate(writer, val_loader, net, criterion, optimizer, epoch, train_args):
    net.eval()

    e1, iou, dice = 0, 0, 0
    iris_e1, iris_dice, iris_iou = 0, 0, 0
    pupil_e1, pupil_dice, pupil_iou = 0, 0, 0
    # iris_hsdf, pupil_hsdf = 0, 0 # calculate hausdorff distance takes too long

    L = len(val_loader)
    for data in val_loader:
        image, mask, iris_edge, iris_mask, pupil_edge, pupil_mask = \
            data['image'], data['mask'], data['iris_edge'], \
            data['iris_edge_mask'], data['pupil_edge'], data['pupil_edge_mask']
   
        image = Variable(image).cuda()
        mask = Variable(mask).cuda()
        iris_edge = Variable(iris_edge).cuda()
        iris_mask = Variable(iris_mask).cuda()
        pupil_edge = Variable(pupil_edge).cuda()
        pupil_mask = Variable(pupil_mask).cuda()
        
        with torch.no_grad():
            outputs = net(image)

        pred_mask, pred_iris_mask, pred_pupil_mask = \
            outputs['pred_mask'], outputs['pred_iris_mask'], outputs['pred_pupil_mask']

        loss_mask = criterion(pred_mask, mask)
        loss_iris = criterion(pred_iris_mask, iris_mask)
        loss_pupil = criterion(pred_pupil_mask, pupil_mask)
        val_loss = loss_mask + loss_iris + loss_pupil        

        pred_iris_circle_mask, pred_iris_edge, _ = get_edge(pred_iris_mask)
        pred_pupil_circle_mask, pred_pupil_egde, _ = get_edge(pred_pupil_mask)        
        #################### val for iris mask ###################
        val_results = evaluate_seg(pred_mask, mask, dataset_name)    
        e1 += val_results['E1']/L
        iou += val_results['IoU']/L
        dice += val_results['Dice']/L

        #################### val for outer edge ##################
        iris_val_results = evaluate_loc(pred_iris_circle_mask, iris_mask, pred_iris_edge, iris_edge, dataset_name)
        iris_e1 += iris_val_results['E1']/L
        iris_dice += iris_val_results['Dice']/L
        iris_iou += iris_val_results['IoU']/L
        # iris_hsdf += iris_val_results['hsdf']/L

        #################### val for inner edge ##################
        pupil_val_results = evaluate_loc(pred_pupil_circle_mask, pupil_mask, pred_pupil_egde, pupil_edge, dataset_name)
        pupil_e1 += pupil_val_results['E1']/L
        pupil_dice += pupil_val_results['Dice']/L  
        pupil_iou += pupil_val_results['IoU']/L
        # pupil_hsdf += pupil_val_results['hsdf']/L
        
    logging.info('------------------------------------------------current val result-----------------------------------------------')    
    logging.info('>maks      epoch:{:2d}   val loss:{:.7f}   E1:{:.7f}   Dice:{:.7f}   IOU:{:.7f}'. \
        format(epoch, loss_mask, e1, dice, iou))
    logging.info('>iris      epoch:{:2d}   val loss:{:.7f}   E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}'. \
        format(epoch, loss_iris, iris_e1, iris_dice, iris_iou))
    logging.info('>pupil     epoch:{:2d}   val loss:{:.7f}   E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}'. \
        format(epoch, loss_pupil, pupil_e1, pupil_dice, pupil_iou))
    
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('e1_val', e1, epoch)
    writer.add_scalar('iou_val', iou, epoch)
    writer.add_scalar('dice_val', dice, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    writer.add_images('image', image, epoch)
    writer.add_images('mask', mask, epoch)
    writer.add_images('pred_mask', pred_mask>0, epoch)
    writer.add_images('iris_mask', iris_mask, epoch)
    writer.add_images('pred_iris_mask', pred_iris_mask>0, epoch)
    writer.add_images('pupil_mask', pupil_mask, epoch)
    writer.add_images('pred_pupil_mask', pred_pupil_mask>0, epoch)

    if e1 < train_args['best_record']['E1']:
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['E1'] = e1
        train_args['best_record']['IoU'] = iou
        train_args['best_record']['Dice'] = dice
        if train_args['gpu_ids']:
            torch.save(net.module.state_dict(), os.path.join(checkpoint_path, 'for_mask.pth'))
        else:
            torch.save(net.state_dict(), os.path.join(checkpoint_path, 'for_mask.pth'))
        checkpoints_name = 'epoch_%d_e1_%.7f_iou_%.7f_dice_%.7f' % (epoch, e1, iou, dice)
        logging.info(f'Saved mask checkpoints {checkpoints_name}.pth!')
    
    if iris_e1 < train_args['best_record_outer']['E1']:
        train_args['best_record_outer']['epoch'] = epoch
        train_args['best_record_outer']['E1'] = iris_e1
        train_args['best_record_outer']['IoU'] = iris_iou
        train_args['best_record_outer']['Dice'] = iris_dice
        if train_args['gpu_ids']:
            torch.save(net.module.state_dict(), os.path.join(checkpoint_path, 'for_mask.pth'))
        else:
            torch.save(net.state_dict(), os.path.join(checkpoint_path, 'for_mask.pth'))
        outer_checkpoints_name = 'epoch_%d_e1_%.7f_iou_%.7f_dice_%.7f' % (epoch, iris_e1, iris_iou, iris_dice)
        logging.info(f'Saved iris checkpoints {outer_checkpoints_name}.pth!')

    if pupil_e1 < train_args['best_record_inner']['E1']:
        train_args['best_record_inner']['epoch'] = epoch
        train_args['best_record_inner']['E1'] = pupil_e1
        train_args['best_record_inner']['IoU'] = pupil_iou
        train_args['best_record_inner']['Dice'] = pupil_dice
        if train_args['gpu_ids']:
            torch.save(net.module.state_dict(), os.path.join(checkpoint_path, 'for_mask.pth'))
        else:
            torch.save(net.state_dict(), os.path.join(checkpoint_path, 'for_mask.pth'))
        inner_checkpoints_name = 'epoch_%d_e1_%.7f_iou_%.7f_dice_%.7f' % (epoch, pupil_e1, pupil_iou, pupil_dice)
        logging.info(f'Saved pupil checkpoints {inner_checkpoints_name}.pth!')

    net.train()
    return val_loss


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    args = get_args()
    train_args = {
        'epoch_num': args.epoch_num,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'checkpoints': args.checkpoints,  # empty string denotes learning from scratch
        'log_name': args.log_name,
        'print_freq': 20,
        'gpu_ids': args.gpu_ids
    }

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    check_mkdir('experiments')
    log_path = os.path.join('experiments', experiment_name + '_' + start_time + '_' + train_args['log_name'].split('.')[0])
    check_mkdir(log_path)
    checkpoint_path = os.path.join(log_path, 'checkpoints')
    check_mkdir(checkpoint_path)
    logging.basicConfig(
        filename=os.path.join(log_path,train_args['log_name']),
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    main(train_args)
