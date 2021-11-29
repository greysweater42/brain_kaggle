import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision
import pickle
from torch.utils.data import Dataset
import torch

INPUT_PATH = Path("input")


class Sartorius(Dataset):

    path = "ds.pickle"

    def __init__(self, cache=True) -> None:
        super(Sartorius, self).__init__()
        self.images, self.masks = self._read_data(cache=cache)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.images[idx], self.masks[idx]
        return self.images[idx], self.masks[idx]

    def subset(self, idx):
        self.images = self.images[idx]
        self.masks = self.masks[idx]

    def _read_data(self, cache=True):
        if cache:
            with open(self.path, "rb") as p:
                return pickle.load(p)
        train = pd.read_csv(INPUT_PATH / "train.csv")
        masks = dict()
        # TODO how about doing this with asyncio, with one task for one id?
        for id in tqdm(set(train.id), desc="transforming annotations"):
            annotations = train.loc[train.id == id, "annotation"].tolist()
            id_masks = []
            for annotation in annotations:
                id_masks.append(self._decodeRLE(annotation))
            mask = np.array(id_masks).sum(0)
            mask[mask > 1] = 1  # overlapping annotations
            masks[id] = torch.tensor(mask)

        ds = dict()
        for id, mask in tqdm(masks.items(), desc="reading images"):
            im = torchvision.io.read_image(str(INPUT_PATH / "train" / f"{id}.png"))
            ds[id] = (im, mask)

        ds = list(ds.values())
        images = torch.stack([i[0] for i in ds])
        masks = torch.stack([i[1] for i in ds])
        data = (images, masks)
        with open(self.path, "wb") as p:
            pickle.dump(data, p)

        return data

    @staticmethod
    def _decodeRLE(annotation):
        size = (520, 704)
        s = annotation.split(" ")
        pixels = []
        for i in range(len(s) // 2):
            start = int(s[2 * i])
            increment = int(s[2 * i + 1])
            for j in range(start, start + increment):
                pixels.append(j - 1)
        im_map = np.zeros(size).flatten()
        im_map[pixels] = 1.0
        im_map = im_map.reshape(size)
        return im_map


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