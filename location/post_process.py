import cv2
import numpy as np
import torch


def get_edge(pred_masks):
    '''
    args:
        pred_mask: tensor(bchw), predicted pupil or iris masks for fit a ellipse
    return:
        pred_circle_masks: solid ellipse mask
        pred_edges: hollow ellipse with sigle pixel
        ellipse_params: tensor(b,5)[[x,y,h,w,r]], ellipse_params
    '''
    batch_size = pred_masks.size()[0]
    pred_masks = np.asarray((pred_masks.cpu()>0).to(dtype=torch.uint8))*255

    pred_circle_masks = np.zeros(pred_masks.shape)
    pred_edges = np.zeros(pred_masks.shape)
    ellipse_params = np.zeros((batch_size, 5))

    for i in range(batch_size):
        contours, hierarchy = cv2.findContours(pred_masks[i][0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            max_area = 0
            max_id = 0
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_contour = contour
                    max_area = area
            ellipse_param = cv2.fitEllipse(max_contour)
            cv2.ellipse(pred_circle_masks[i, 0, :, :], ellipse_param, (255,255,255), -1)
            cv2.ellipse(pred_edges[i, 0, :, :], ellipse_param, (255,255,255), 1)
            ellipse_params[i] = [*ellipse_param[0], *ellipse_param[1], ellipse_param[2]] #[x,y,h,w,r]

        except:
            pass

    pred_circle_masks = torch.tensor(pred_circle_masks).cuda()
    pred_edges = torch.tensor(pred_edges).cuda()
    ellipse_params = torch.tensor(ellipse_params).cuda()
    
    return pred_circle_masks, pred_edges, ellipse_params

