import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision
import pickle
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

import torch



INPUT_PATH = Path("input")


class Sartorius(Dataset):

    path = "ds.pickle"

    def __init__(self, cache=True, transform=None) -> None:
        super(Sartorius, self).__init__()
        self.images, self.masks = self._read_data(cache=cache)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            seed = torch.seed()
            imgs = self.transform(self.images[idx])
            torch.manual_seed(seed)
            masks = self.transform(self.masks[idx])
            return imgs, masks
        return self.images[idx], self.masks[idx]

    def subset(self, idx):
        self.images = self.images[idx]
        self.masks = self.masks[idx]

    @staticmethod
    def _concat_masks(x):
        mask = x.to_numpy().sum(0)
        mask[mask > 1] = 1  # overlapping annotations
        return mask.astype(np.int64)

    def _read_data(self, cache=True):
        if cache:
            with open(self.path, "rb") as p:
                return pickle.load(p)
        train = pd.read_csv(INPUT_PATH / "train.csv")
        train["mask"] = train["annotation"].map(
            lambda x: rle_decode(x, shape=(520, 704))
        )

        masks = train.groupby("id", as_index=False).agg({"mask": self._concat_masks})
        ds = []
        for _, (id, mask) in tqdm(masks.iterrows(), desc="reading images"):
            im = torchvision.io.read_image(str(INPUT_PATH / "train" / f"{id}.png"))
            ds.append([im, torch.tensor(mask)])

        images = torch.stack([i[0] for i in ds])
        masks = torch.stack([i[1] for i in ds])
        data = (images, masks)
        with open(self.path, "wb") as p:
            pickle.dump(data, p)

        return data


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def encodeRLE(p, labels):
    res = []
    is_1 = False
    c = 0
    for i, (v, l) in enumerate(zip(p.flatten(), labels.flatten())):
        if v == 1:
            c += 1
            if not is_1:
                res.append(l)  # label
                res.append(i)  # num of starting pixel
                is_1 = True
        if v == 0:
            if c != 0:
                res.append(c)  # length
            c = 0
            is_1 = False
    assert not len(res) % 3
    return res


def make_predictions_for_test_set(net, device):
    import pandas as pd
    import cv2
    from collections import defaultdict

    net.eval()
    preds = []
    for image in Path(INPUT_PATH / "test").glob("*.png"):
        im = torchvision.io.read_image(str(image))
        pred = net(im.unsqueeze(0).to(device, dtype=torch.float32))
        p = pred.detach()[0].argmax(0)
        p = np.array(p).astype(np.uint8)
        _, labels = cv2.connectedComponents(np.array(p))
        p_e = encodeRLE(p, labels)
        labels = p_e[::3]
        starts = p_e[1::3]
        lengths = p_e[2::3]
        d = defaultdict(lambda: [])
        for l, s, le in zip(labels, starts, lengths):
            d[l] += [s, le]
        rles = [" ".join([str(p) for p in d]) for d in d.values()]
        rows = []
        for rle in rles:
            ord = int(rle.split(" ")[0])
            rows.append(dict(id=image.stem, predicted=rle, ord=ord))
        preds += rows
    return pd.DataFrame(preds).sort_values(["id", "ord"])[["id", "predicted"]]
