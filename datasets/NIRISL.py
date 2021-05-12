import os
import random
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# root = '.../NIRISL-dataset/' # change this as you need

def make_dataset_list(dataset_name, mode, val_percent=0.2):

    root_train = root + 'train/'
    root_test = root + 'test/'

    assert dataset_name in ['CASIA-Iris-Africa','CASIA-distance', 'Occlusion', 'Off_angle', 'CASIA-Iris-Mobile-V1.0']
    assert mode in ['train', 'val', 'test']
    dataset_name = correct_dataset_name(dataset_name)

    if mode == 'test':
        data_path = {'images_path': os.path.join(root_test, dataset_name, 'image')}
        test_filenames_list = list(os.listdir(data_path['images_path']))
        return data_path, test_filenames_list

    data_path = {
        'images_path': os.path.join(root_train, dataset_name, 'image'),
        'masks_path': os.path.join(root_train, dataset_name, 'SegmentationClass'),
        'irises_edge_path': os.path.join(root_train, dataset_name, 'iris_edge'),
        'irises_edge_mask_path': os.path.join(root_train, dataset_name, 'iris_edge_mask'),
        'pupils_edge_path': os.path.join(root_train, dataset_name, 'pupil_edge'),
        'pupils_edge_mask_path': os.path.join(root_train, dataset_name, 'pupil_edge_mask')
    }

    images_filenames_list = list(os.listdir(data_path['images_path']))
    random.seed(42)
    random.shuffle(images_filenames_list)
    train_filenames_list = images_filenames_list[:int((1-val_percent)*len(images_filenames_list))]
    val_filenames_list = images_filenames_list[int((1-val_percent)*len(images_filenames_list)):]

    if mode == 'train':
        return data_path, train_filenames_list
    else:
        return data_path, val_filenames_list


def correct_dataset_name(dataset_name):
    if dataset_name == 'CASIA-distance':
        return 'CASIA-Iris-Asia/CASIA-distance'
    elif dataset_name == 'Occlusion':
        return 'CASIA-Iris-Asia/CASIA-Iris-Complex/Occlusion'
    elif dataset_name == 'Off_angle':
        return 'CASIA-Iris-Asia/CASIA-Iris-Complex/Off_angle'
    else:
        return dataset_name


def get_heatmap(iris_edge, pupil_edge):
    '''input and output are numpy array'''
    iris_edge_blur = cv2.GaussianBlur(iris_edge, (55,55), 0)
    pupil_edge_blur = cv2.GaussianBlur(pupil_edge, (27,27), 0)
    iris_edge_blur = iris_edge_blur / np.max(iris_edge_blur)
    pupil_edge_blur = pupil_edge_blur / np.max(pupil_edge_blur)
    loc_blur = ((iris_edge_blur + pupil_edge_blur) / np.max(iris_edge_blur + pupil_edge_blur)) * 255
    return loc_blur.astype(np.uint8)


class nirislDataset(Dataset):
    '''
    args:
        dataset_name(str): support for 'CASIA-Iris-Africa','CASIA-distance', 'Occlusion', 'Off_angle', 'CASIA-Iris-Mobile-V1.0'
        mode(str): 'train', 'val', 'test'
        transform(dict): {'train': train_augment, 'test': test_augment}

    return(dict): {
        'image': aug_img,
        'mask': aug_mask,
        'iris_edge': aug_iris_edge
        'iris_edge_mask': aug_iris_edge_mask
        'pupil_edge': aug_pupil_edge
        'pupil_edge_mask': aug_pupil_edge_mask
    }
    '''
    def __init__(self, dataset_name, mode, transform=None, val_percent=0.2):
        self.dataset_name = dataset_name
        self.mode = mode
        self.transform = transform
        self.data_path, self.data_list = make_dataset_list(dataset_name, mode, val_percent=val_percent)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        if self.mode == 'test':
            image_name = self.data_list[idx].split('.')[0]
            image = Image.open(os.path.join(self.data_path['images_path'], self.data_list[idx]))
            
            if self.transform is not None:
                image = np.asarray(image)
                aug_data = self.transform(image=image)
                aug_image = aug_data['image']
                image = Image.fromarray(aug_image)

            image = transforms.ToTensor()(image)
            return {
                'image_name': image_name,
                'image': image
            }
        
        image_name = self.data_list[idx].split('.')[0]
        image = Image.open(os.path.join(self.data_path['images_path'], self.data_list[idx]))
        mask = Image.open(os.path.join(self.data_path['masks_path'], image_name + '.png'))
        iris_edge = Image.open(os.path.join(self.data_path['irises_edge_path'], image_name + '.png'))
        iris_edge_mask = Image.open(os.path.join(self.data_path['irises_edge_mask_path'], image_name + '.png'))
        pupil_edge = Image.open(os.path.join(self.data_path['pupils_edge_path'], image_name + '.png'))
        pupil_edge_mask = Image.open(os.path.join(self.data_path['pupils_edge_mask_path'], image_name + '.png'))

        if self.transform is not None:
            image = np.asarray(image)
            mask = np.asarray(mask)
            iris_edge = np.asarray(iris_edge)
            iris_edge_mask = np.asarray(iris_edge_mask)
            pupil_edge = np.asarray(pupil_edge)
            pupil_edge_mask = np.asarray(pupil_edge_mask)
            heatmap = get_heatmap(iris_edge, pupil_edge)
            mask_list = [mask, iris_edge, iris_edge_mask, pupil_edge, pupil_edge_mask, heatmap]

            aug_data = self.transform(image=image, masks=mask_list)
            aug_image, aug_mask_list = aug_data['image'], aug_data['masks']
            
            image = Image.fromarray(aug_image)
            mask = Image.fromarray(aug_mask_list[0])
            iris_edge = Image.fromarray(aug_mask_list[1])
            iris_edge_mask = Image.fromarray(aug_mask_list[2])
            pupil_edge = Image.fromarray(aug_mask_list[3])
            pupil_edge_mask = Image.fromarray(aug_mask_list[4])
            heatmap = Image.fromarray(aug_mask_list[5])

        aug_image = transforms.ToTensor()(image)
        aug_mask = transforms.ToTensor()(mask)
        aug_iris_edge = transforms.ToTensor()(iris_edge)
        aug_iris_edge_mask = transforms.ToTensor()(iris_edge_mask)
        aug_pupil_edge = transforms.ToTensor()(pupil_edge)
        aug_pupil_edge_mask = transforms.ToTensor()(pupil_edge_mask)
        aug_heatmap = transforms.ToTensor()(heatmap)

        return {
            'image_name': image_name,
            'image': aug_image,
            'mask': aug_mask,
            'iris_edge': aug_iris_edge,
            'iris_edge_mask': aug_iris_edge_mask,
            'pupil_edge': aug_pupil_edge,
            'pupil_edge_mask': aug_pupil_edge_mask,
            'heatmap': aug_heatmap
        }

