import pandas as pd
from IPython.display import display
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

path = Path("input")
train = pd.read_csv(path / "train.csv")


def decodeRLE(annotation, im):
    size = (im.size[1], im.size[0])
    s = annotation.split(" ")
    pixels = []
    for i in range(len(s) // 2):
        start = int(s[2 * i])
        increment = int(s[2 * i + 1])
        for j in range(start, start + increment):
            pixels.append(j-1)
    im_map = np.zeros(size).flatten()
    im_map[pixels] = 1.0
    im_map = im_map.reshape(size)
    return im_map


def draw_annotations(id: str, num: int = None):
    if not num:
        num = len(train[train.id == id])
    with Image.open(path / "train" / f"{id}.png") as im:
        annotations = [decodeRLE(a, im) for a in train[train.id == id].annotation[:num]]
        im = im.convert("RGBA")
        overlay = Image.new("RGBA", im.size)
        draw = ImageDraw.Draw(overlay)
        for annotation in annotations:
            for i in range(annotation.shape[0]):
                for j in range(annotation.shape[1]):
                    if annotation[i, j]:
                        draw.point((j, i), fill=(127, 0, 0, 50))
        im = Image.alpha_composite(im, overlay)
        display(im)


id = np.unique(train.id)[2]
draw_annotations(id, 100)
