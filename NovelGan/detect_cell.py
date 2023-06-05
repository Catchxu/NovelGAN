import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import anndata as ad
from typing import Dict

from ._pretrain import Pretrain
from .generator import Memory_G
from .discriminator import Discriminator
from ._utils import seed_everything


def Detect_cell(train: ad.AnnData,
                test: ad.AnnData,
                n_epochs: int = 50,
                batch_size: int = 128,
                learning_rate: float = 1e-4,
                mem_dim: int = 2048,
                GPU: bool = True,
                verbose: bool = True,
                log_interval: int = 10,
                random_state: int = None,
                weight: Dict = None,
                pre_train: bool = True):

    if GPU:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU isn't available, and use CPU to train NovelGAN.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if random_state is not None:
        seed_everything(random_state)

    if (train.var_names != test.var_names).any():
        raise RuntimeError('Train data and test data have different genes.')
    n_gene = train.n_vars
    train = train.X
    test = test.X

    # Initialize dataloader for train data
    train = torch.as_tensor(torch.from_numpy(train), dtype=torch.float32)
    test = torch.as_tensor(torch.from_numpy(test), dtype=torch.float32)
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test,
                             batch_size=batch_size*5,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    # Initialize Discriminator and Generator
    D = Discriminator(n_gene).to(device)
    G = Memory_G(n_gene).to(device)
    if pre_train:
        path_1 = './pretrain/Encoder.pth'
        path_2 = './pretrain/Decoder.pth'
        if not (os.path.exists(path_1) and os.path.exists(path_2)):
            Pretrain(train)

        # load the pre-trained weights for Encoder and Decoder
        pre_weights = torch.load(path_1)
        G.Encoder1.load_state_dict({k: v for k, v in pre_weights.items()})
        G.Encoder2.load_state_dict({k: v for k, v in pre_weights.items()})
        pre_weights = torch.load(path_2)
        G.Decoder.load_state_dict({k: v for k, v in pre_weights.items()})

    opt_D = optim.Adam(D.parameters(),
                       lr=learning_rate,
                       betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(),
                       lr=learning_rate*2,
                       betas=(0.5, 0.999))
    D_scaler = torch.cuda.amp.GradScaler()
    G_scaler = torch.cuda.amp.GradScaler()
    L1 = nn.L1Loss().to(device)
    L2 = nn.MSELoss().to(device)
    BCE = nn.BCELoss().to(device)

    # Initialize the weight of generator's loss
    if weight is None:
        weight = {'w_enc': 1, 'w_rec': 50, 'w_adv': 1}

    D.train()
    G.train()
    # train the NovelGAN
    for epoch in range(n_epochs):
        for idx, data in enumerate(train_loader):
            data = data.to(device)

            # update G-Net
            real_z, fake_data, fake_z = G(data)
            real_d = torch.sigmoid(D(data))
            fake_d = torch.sigmoid(D(fake_data))

            Loss_enc = L2(real_z, fake_z)
            Loss_rec = L1(data, fake_data)
            Loss_adv = L2(real_d, torch.zeros_like(real_d)) + L2(fake_d, torch.ones_like(fake_d))
            G_loss = (weight['w_enc']*Loss_enc +
                      weight['w_rec']*Loss_rec +
                      weight['w_adv']*Loss_adv)

            opt_G.zero_grad()
            G_scaler.scale(G_loss).backward(retain_graph=True)
            G_scaler.step(opt_G)
            G_scaler.update()

            # update D-Net
            real_z, fake_data, fake_z = G(data)
            real_d = torch.sigmoid(D(data))
            fake_d = torch.sigmoid(D(fake_data.detach()))

            D_loss = BCE(real_d, torch.ones_like(real_d)) + BCE(fake_d, torch.zeros_like(fake_d))

            opt_D.zero_grad()
            D_scaler.scale(D_loss).backward()
            D_scaler.step(opt_D)
            D_scaler.update()

        if verbose and ((epoch+1) % log_interval == 0):
            print('Train Epoch: [{}/{} ({:.0f}%)]\tG_loss: {:.6f}\tD_loss: {:.6f}'.format(
                   epoch+1, n_epochs, 100.*(epoch+1)/n_epochs, G_loss.item(), D_loss.item()))

    # test the NovelGAN and detect novel cell type
    G.eval()
    with torch.no_grad():
        diff = torch.empty((0, 1)).to(device)
        for idx, data in enumerate(test_loader):
            data = data.to(device)
            real_z, fake_test, fake_z = G(data)
            d = 1 - F.cosine_similarity(real_z, fake_z).reshape(-1, 1)
            diff = torch.cat([diff, d], dim=0)

    diff = diff.cpu().numpy()
    diff = (diff - diff.min())/(diff.max() - diff.min())
    return diff.reshape(-1)
