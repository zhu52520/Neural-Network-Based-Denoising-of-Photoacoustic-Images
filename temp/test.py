import scipy.io as scio
import numpy as np
import math
import os
import cv2
from matplotlib import pyplot as plt
from skimage import metrics

def ssim(x,y):
    return metrics.structural_similarity(x,y)

def psnr(x,y):
    mse = np.mean((x-y)**2)
    psnr_result = 10*math.log10(1/mse) #针对浮点型数据，最大像素值为 1
    return psnr_result

def read_mat(index):
    noise_file = "brain" + str(index) + "_X.mat"
    noise_dir = os.path.join("E:\\文件\\大学\\大二上\\模拟电路\\project\\project_data\\Signal_withnoise", noise_file)
    X = scio.loadmat(noise_dir)
    truth_file = "brain" + str(index) + "_Y.mat"
    truth_dir = os.path.join("E:\\文件\\大学\\大二上\\模拟电路\\project\\project_data\\Ground_truth", truth_file)
    Y = scio.loadmat(truth_dir)
    return {**X,**Y}

def draw(result, noise, truth):
    fig=plt.figure(num=1,figsize=(4,4))
    ax1=fig.add_subplot(221)
    ax1.imshow(result,aspect='auto')
    ax1.set_title("result")
    ax2=fig.add_subplot(222)
    ax2.imshow(noise,aspect='auto')
    ax2.set_title("signal_withnoise")
    ax3=fig.add_subplot(223)
    ax3.imshow(truth,aspect='auto')
    ax3.set_title("ground_truth")
    plt.show()

a = read_mat(1)
temp = a['X']-a['Y']
result = a['X']-temp
print(psnr(result,a['Y']))
