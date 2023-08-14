import torch
import torch.nn as nn
import numpy as np
from monai.inferers import sliding_window_inference
from pietorch.DuRN_P_no_norm import cleaner as cleaner
from skimage.metrics import peak_signal_noise_ratio as psnr

def inference(
    device,
    test_loader,
    model_save_path,
    roi_size
    ):


    assert test_loader.batch_size == 1

    net = cleaner().to(device)
    net.load_state_dict(torch.load(model_save_path, device))
    net.eval()

    with torch.no_grad():
        results = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = sliding_window_inference(
                inputs=x,
                roi_size=roi_size,
                sw_batch_size=2,
                predictor=net,
                overlap=0,
                mode='gaussian'
                )

            x, y, y_pred = (
                np.array(x.cpu()).squeeze(),
                np.array(y.cpu()).squeeze(),
                np.array(y_pred.cpu()).squeeze()
                )

            psnr_score = psnr(y, y_pred, data_range=1)
            results.append((x, y, y_pred, psnr_score))
    print('inference completed.')
    return results
    

