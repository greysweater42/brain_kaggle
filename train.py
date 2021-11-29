from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from dataset import Sartorius
import numpy as np

np.random.seed(42)


train = Sartorius()
val = Sartorius()
val_idxs = np.random.choice(range(len(train)), 100, replace=False)
train_idxs = set(range(len(train))) - set(val_idxs)
train.subset(list(train_idxs))
val.subset(val_idxs)

device = "cuda"
net = UNet(n_channels=1, n_classes=2, bilinear=True)
net.to(device=device)

epochs = 5
batch_size = 1
learning_rate = 0.001

loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
train_loader = DataLoader(train, shuffle=True, **loader_args)
val_loader = DataLoader(val, shuffle=True, **loader_args)

optimizer = optim.RMSprop(
    net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
criterion = nn.CrossEntropyLoss()

# net.load_state_dict(torch.load("model.torch"))

def dice_score(y, y_hat):
    TP = ((y == 1) & (y_hat == 1)).sum()
    FP = ((y == 0) & (y_hat == 1)).sum()
    FN = ((y == 1) & (y_hat == 0)).sum()
    dice = 2 * TP / (2 * TP + FP + FN)
    return dice


train_losses = []
train_scores = []
val_scores = []
for i in range(epochs):
    epoch_loss = 0
    net.train()
    for images, true_masks in tqdm(train_loader, desc=f"tr tr Epoch: {i}"):
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        masks_pred = net(images)
        loss = criterion(masks_pred, true_masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss)
    
    net.eval()
    dice_scores = 0
    for images, true_masks in tqdm(train_loader, desc=f"tr va Epoch: {i}"):
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        masks_pred = net(images)
        dice_scores += dice_score(true_masks[0], masks_pred.detach()[0].argmax(0))
    train_scores.append(dice_scores / len(train))

    dice_scores = 0
    for images, true_masks in tqdm(val_loader, desc=f"va va Epoch: {i}"):
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        masks_pred = net(images)
        dice_scores += dice_score(true_masks[0], masks_pred.detach()[0].argmax(0))
    val_scores.append(dice_scores / len(val))


torch.save(net.state_dict(), "/kaggle/working/model.torch")