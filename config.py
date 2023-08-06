# -*- coding: utf-8 -*-
"""
configuration for model
"""

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 64
LEARN_RATE = 1e-4