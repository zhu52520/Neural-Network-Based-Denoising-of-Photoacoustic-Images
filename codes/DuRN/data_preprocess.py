import os
from glob import glob
import numpy as np
import scipy.io as scio
import shutil

from configs import root_dir


def from_mat_to_numpy(mat_path:str, key:str) -> np.ndarray:
    return scio.loadmat(mat_path)[key].astype('float32')


preproc_dir = os.path.join(root_dir, "preprocessed")

if os.path.exists(preproc_dir):
    shutil.rmtree(preproc_dir, ignore_errors=True)

preproc_noise_dir = os.path.join(preproc_dir, 'noise')
preproc_gt_origin_dir = os.path.join(preproc_dir, 'gt_origin')
os.makedirs(preproc_noise_dir)
os.makedirs(preproc_gt_origin_dir)


raw_noise_paths = sorted(glob(os.path.join(root_dir, 'Signal_withnoise', 'brain*_X.mat')))
raw_gt_paths = sorted(glob(os.path.join(root_dir, 'Ground_truth', 'brain*_Y.mat')))

preproc_noises = list(map(lambda path:from_mat_to_numpy(path, 'X'), raw_noise_paths))
preproc_gts = list(map(lambda path:from_mat_to_numpy(path, 'Y'), raw_gt_paths))

for idx, (x, y) in enumerate(zip(preproc_noises, preproc_gts)):
    np.save(os.path.join(preproc_noise_dir, f'brain{idx+1}_X.npy'), x)
    np.save(os.path.join(preproc_gt_origin_dir, f'brain{idx+1}_Y.npy'), y)

print('data processed, succeed.')


