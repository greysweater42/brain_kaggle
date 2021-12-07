# %%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Sartorius
import numpy as np
import torchvision.transforms as transforms


t = transforms.RandomCrop((256, 256))
train = Sartorius(transform=t)

# %%
loader_args = dict(batch_size=1, num_workers=2, pin_memory=True)
train_loader = DataLoader(train, shuffle=True, **loader_args)

# %%
from PIL import Image
import requests
from io import BytesIO

url = "https://earthsky.org/upl/2018/06/ocean-apr27-2020-Cidy-Chai-North-Pacific-scaled-e1591491800783.jpeg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))


# %%
import segmentation_models_pytorch as smp
import torch

device = "cuda"
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
model.to(device)

images = im.to(device=device, dtype=torch.float32)
# %%
images = images.unsqueeze(0)
images
model(images).shape


from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
