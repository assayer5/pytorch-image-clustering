# -*- coding: utf-8 -*-
"""
define model 
"""

import torch
import torch.nn as nn

# define cnn model 
class cnn_autoencoder(nn.Module):
    def __init__(self):
        super(cnn_autoencoder, self).__init__()
        
        self.act = nn.ReLU()
        self.flat = nn.Flatten()
        self.sig = nn.Sigmoid()

        self.c1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(32, 64, 7)

        self.t1 = nn.ConvTranspose2d(64, 32, 7)
        self.t2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.t3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, image):
        # encode image
        img = self.act(self.c1(image))
        img = self.act(self.c2(img))
        img = self.flat(self.c3(img))

        # decode image
        img = img.view(-1, 64, 1, 1)
        img = self.act(self.t1(img))
        img = self.act(self.t2(img))
        img = self.sig(self.t3(img))
        
        return img