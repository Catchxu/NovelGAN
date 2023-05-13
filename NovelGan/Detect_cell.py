import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict

from .generator import Generator
from .discriminator import Discriminator


def Detect_cell(train: np.array,
                test: np.array,
                n_epochs: int = 100,
                batch_size: int = 32,
                learning_rate: float = 0.0005,
                mem_dim: int = 2048,
                GPU: bool = True,
                verbose: bool = True,
                log_interval: int = 10,
                random_state: int = 100,
                weight: Dict = None):
    '''
    This function is for novel cell type detection via NovelGan.

    Parameters
    ----------
    train: array
        The train expression matrix of shape n_obs x n_var.
        Rows correspond to cells and columns to genes.
        The cells in train will be regarded as known cell type.
    test: array
        The test expression matrix of shape n_obs x n_var.
        Rows correspond to cells and columns to genes.
        It includes some novel cells which are different from train
    n_epochs: int
        The numbers of epochs to train NovelGan.
    batch_size: int
        The batch size of train datasets.
    learning_rate: float
        The learning rate of the generator and discriminator's optimizer.
    mem_dim: int
        The size of memory bank.
    GPU: bool
        If 'True', the NovelGan will be trained on GPU (if possible).
    verbose: bool
        If 'True", prints the details in train.
    log_interval: int
        The interval epochs between two prints of training information.
    random_state: int
        Change to use different initial states for the optimization.
    weight: dictionary
        The weight of every parts loss of generator's loss.
        e.g. {'w_enc': 1, 'w_rec': 30, 'w_adv': 1}
    
    Returns
    -------
    
    '''

    if GPU:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU isn't available, and use CPU to train NovelGan.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if random_state is not None:
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)
        random.seed(random_state)
        torch.backends.cudnn.deterministic = True

    # Initialize dataloader for train data
    train = torch.as_tensor(torch.from_numpy(train), dtype=torch.float32)
    train = train.reshape(train.shape[0], 1, -1)
    test = torch.as_tensor(torch.from_numpy(test), dtype=torch.float32)
    test = test.reshape(test.shape[0], 1, -1)
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test,
                             batch_size=batch_size * 5,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    # Initialize Discriminator and Generator
    z_dim = int((train.shape[2]/8)) - 1
    D = Discriminator().to(device)
    G = Generator(1, 64, mem_dim, z_dim).to(device)

    opt_D = optim.Adam(D.parameters(),
                       lr=learning_rate,
                       betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(),
                       lr=learning_rate,
                       betas=(0.5, 0.999))

    D_scaler = torch.cuda.amp.GradScaler()
    G_scaler = torch.cuda.amp.GradScaler()
    L1 = nn.L1Loss().to(device)
    L2 = nn.MSELoss().to(device)
    BCE = nn.BCELoss().to(device)

    # Initialize the weight of generator's loss
    if weight is None:
        weight = {'w_enc': 1, 'w_rec': 32, 'w_adv': 1}

    D.train()
    G.train()
    # train the NovelGan
    for epoch in range(n_epochs):
        for idx, data in enumerate(train_loader):
            data = data.to(device)

            # forward
            real_z, fake_data, fake_z = G(data)
            real_d = D(data)
            fake_d = D(fake_data.detach())

            # update G-Net
            Loss_enc = L2(real_z, fake_z)
            Loss_rec = L1(data, fake_data)
            real_loss = L2(real_d, torch.ones_like(real_d))
            fake_loss = L2(fake_d, torch.zeros_like(fake_d))
            Loss_adv = real_loss + fake_loss
            G_loss = (weight['w_enc']*Loss_enc +
                      weight['w_rec']*Loss_rec +
                      weight['w_adv']*Loss_adv)

            opt_G.zero_grad()
            G_scaler.scale(G_loss).backward(retain_graph=True)
            G_scaler.step(opt_G)
            G_scaler.update()

            # update D-Net
            real_loss = BCE(real_d, torch.ones_like(real_d))
            fake_loss = BCE(fake_d, torch.zeros_like(fake_d))
            D_loss = real_loss + fake_loss

            opt_D.zero_grad()
            D_scaler.scale(D_loss).backward()
            D_scaler.step(opt_D)
            D_scaler.update()

            # update the memory bank finally
            G.Memory.update_mem(real_z)

        if verbose and ((epoch+1) % log_interval == 0):
            print('Train Epoch: [{}/{} ({:.0f}%)]\tG_loss: {:.6f}\tD_loss: {:.6f}'.format(
                   epoch+1, n_epochs, 100.*(epoch+1)/n_epochs, G_loss.item(), D_loss.item()))

    # test the NovelGan and detect novel cell type
    G.eval()
    with torch.no_grad():
        diff = torch.empty((0, real_z.shape[2])).to(device)
        for idx, data in enumerate(test_loader):
            data = data.to(device)
            real_z, fake_test, fake_z = G(data)
            d = nn.MSELoss(reduce=False)(real_z, fake_z)
            d = d.reshape(d.shape[0], d.shape[2])
            diff = torch.cat([diff, d], dim=0)

    diff = diff.cpu().numpy()

    return diff
