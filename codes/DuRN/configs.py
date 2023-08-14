import os
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.data import ArrayDataset
from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage,
    RandSpatialCrop,
    EnsureType
)




# change devices regarding to your machine
device = torch.device('cuda:0')

# where your data root is
root_dir = 'E:\\文件\\大学\\大二上\\模拟电路\\project\\project_data'

# data filenames, sorted to match
noises = sorted(glob(os.path.join(root_dir, 'preprocessed/noise', 'brain*_X.npy')))
gt_origins = sorted(glob(os.path.join(root_dir, 'preprocessed/gt_origin', 'brain*_Y.npy')))
gt_residues = sorted(glob(os.path.join(root_dir, 'preprocessed/gt_residue', 'brain*_R.npy')))


# data preprocessing

roi_size = (256, 1200)

train_noisetrans = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        RandSpatialCrop(roi_size, random_size=False),
        EnsureType(),
    ]
)
train_gttrans = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        RandSpatialCrop(roi_size, random_size=False),
        EnsureType(),
    ]
)

test_noisetrans = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        EnsureType(),
    ]
)
test_gttrans = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        EnsureType(),
    ]
)

# dataset & dataloader
train_data = ArrayDataset(img=noises[:6], img_transform=train_noisetrans, seg=gt_origins[:6], seg_transform=train_gttrans)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)

test_data = ArrayDataset(img=noises[6:], img_transform=test_noisetrans, seg=gt_origins[6:], seg_transform=test_gttrans)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)



model_save_path = '.\\best_trained_model.pth'


# args to pass to train()
train_args = {
    'device': device,
    'train_loader': train_loader,
    'model_save_path': model_save_path,
    'max_epoch': 100,
    'ssim_weight': 1,
    'l1_loss_weight': 10,
    'base_lr': 5e-5,
    'weight_decay': 1e-8
}


inference_args = {
    'device': device,
    'test_loader': test_loader,
    'model_save_path': model_save_path,
    'roi_size': roi_size
}