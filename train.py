# %%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Sartorius
import numpy as np
import torchvision.transforms as transforms


train = Sartorius()

im, l = train[0]
tr(im).shape
tr(l)
l

# https://github.com/pytorch/vision/issues/9#issuecomment-789308878
t = transforms.RandomRotation(degrees=360)
state = torch.get_rng_state()
x = t(x)
torch.set_rng_state(state)
y = t(y)

tr = transforms.RandomCrop((256, 256))
loader_args = dict(batch_size=1, num_workers=2, pin_memory=True)
train_loader = DataLoader(train, shuffle=True, , transforms=tr, **loader_args)
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
tr = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

tr(img)

im = tr(img)
images = im.to(device=device, dtype=torch.float32)
# %%
images = images.unsqueeze(0)
images
model(images).shape


images
images = torch.cat((torch.zeros((1, 1, 10, 704)).to("cuda"), images), dim=2)
images.shape

images.shape
model(images[:,:,:,:700])
enc = model.encoder(images)

images[:,:,:519].shape

from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
preprocess_input(images[0].to("cpu"))
