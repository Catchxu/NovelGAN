import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import anndata as ad
from typing import Dict

from ._pretrain import Pretrain
from .generator import Align_G
from .discriminator import Discriminator
from ._utils import seed_everything, calculate_gradient_penalty


class Align_cell:
    def __init__(self, n_epochs: int = 2000, learning_rate: float = 5e-4,
                 GPU: bool = True, verbose: bool = True, log_interval: int = 400,
                 random_state: int = None, pre_train: bool = True, weight: Dict = None):
        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train NovelGAN.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if random_state is not None:
            seed_everything(random_state)

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.log_interval = log_interval
        self.pre_train = pre_train

        if weight is None:
            self.weight = {'w_rec': 50, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight

    def fit(self, input: ad.AnnData, base: ad.AnnData):
        if (base.var_names != input.var_names).any():
            raise RuntimeError('base data and input data have different genes.')

        self.n_gene = base.n_vars
        self.base = torch.as_tensor(torch.from_numpy(base.X), dtype=torch.float32).to(self.device)
        self.input = torch.as_tensor(torch.from_numpy(input.X), dtype=torch.float32).to(self.device)

        self.D = Discriminator(in_dim=256, out_dim=[128, 64, 1]).to(self.device)
        self.G = Align_G(base.shape[0], input.shape[0], self.n_gene).to(self.device)

        if self.pre_train:
            self.load_weight()  # load the pre-trained model for better train processes

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()
        self.L1 = nn.L1Loss().to(self.device)
        self.L2 = nn.MSELoss().to(self.device)

        self.train()  # begin to train NovelGAN

        fake_z, z, match = self.G(self.input, self.base)
        match = F.relu(match).detach().cpu().numpy()
        idx = pd.DataFrame({'base_idx': base.obs.index,
                            'input_idx': input.obs.index[match.argmax(axis=1)]})
        return idx

    def load_weight(self):
        path = './pretrain/Encoder.pth'
        if not os.path.exists(path):
            Pretrain(self.base)

        pre_weights = torch.load(path)  # load the pre-trained weights for Encoder
        self.G.Encoder.load_state_dict({k: v for k, v in pre_weights.items()})

        # freeze the encoder weights
        for name, value in self.G.named_parameters():
            if 'Encoder' in name:
                value.requires_grad = False

    def train(self):
        self.D.train()
        self.G.train()
        for epoch in range(self.n_epochs):
            self.G_forward()
            self.G_backward()
            self.D_forward()
            self.D_backward()

            if self.verbose:
                self.log_print(epoch)

    def G_forward(self):
        fake_z, z, match = self.G(self.input, self.base)
        real_d = self.D(z)
        fake_d = self.D(fake_z)

        Loss_rec = self.L1(fake_z, z)
        real_d = torch.sigmoid(real_d)
        fake_d = torch.sigmoid(fake_d)
        Loss_adv = (self.L2(real_d, torch.zeros_like(real_d))+
                    self.L2(fake_d, torch.ones_like(fake_d)))
        self.G_loss = (self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

    def G_backward(self):
        self.opt_G.zero_grad()
        self.G_scaler.scale(self.G_loss).backward()
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()

    def D_forward(self):
        fake_z, z, match = self.G(self.input, self.base)
        real_d = self.D(z)
        fake_d = self.D(fake_z.detach())

        # Compute W-div gradient penalty
        gp = calculate_gradient_penalty(z, fake_z, self.D)
        self.D_loss = -torch.mean(real_d) + torch.mean(fake_d) + gp*self.weight['w_gp']

    def D_backward(self):
        self.opt_D.zero_grad()
        self.D_scaler.scale(self.D_loss).backward()
        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()

    def log_print(self, epoch):
        if (epoch+1) % self.log_interval == 0:
            txt = 'Epoch:[{:^4}/{:^4}({:^3.0f}%)]\t\tG_loss:{:.6f}\t\tD_loss:{:.6f}'
            txt = txt.format(epoch+1, self.n_epochs, 100.*(epoch+1)/self.n_epochs,
                             self.G_loss.item(), self.D_loss.item())
            print(txt)