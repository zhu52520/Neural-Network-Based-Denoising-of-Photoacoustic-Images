import os
import shutil

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from pietorch.DuRN_P_no_norm import cleaner as cleaner
from pietorch.pytorch_ssim import ssim as ssim
from tqdm.notebook import  tqdm


def train_model(
    device,
    train_loader,
    model_save_path,
    max_epoch,
    ssim_weight,
    l1_loss_weight,
    base_lr,
    weight_decay,
    ):

    # clear workspace
    if os.path.exists('./logs'):
        shutil.rmtree('./logs', ignore_errors=True)

    # training mode
    net = cleaner().to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, weight_decay=weight_decay)
    L1_loss = nn.L1Loss()

    epoch_len = len(train_loader)

    writer = SummaryWriter(log_dir='./logs')

    # run for each epoch
    for epoch in tqdm(range(max_epoch)):
        epoch_loss = 0
        step = 0

        for x, y in train_loader:
            step += 1
            # return x, y
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(x)

            ssim_loss = -ssim(outputs, y)
            ssim_loss = ssim_loss*ssim_weight

            l1_loss   = L1_loss(outputs, y)
            l1_loss   = l1_loss*l1_loss_weight

            loss = ssim_loss + l1_loss

            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        print(f"epoch {epoch + 1}/{max_epoch}, average loss: {epoch_loss:.4f}")

        if epoch==10:
            for param_group in optimizer.param_groups:
                param_group['lr']*= 10
        if epoch==40 or epoch==70:
            for param_group in optimizer.param_groups:
                param_group['lr']*= 0.2
                l1_loss_weight *= 0.3

    writer.close()

    # save the model parameters
    torch.save(net.state_dict(), model_save_path)

    print('training completed.')

if __name__ == "__main__":
    pass

