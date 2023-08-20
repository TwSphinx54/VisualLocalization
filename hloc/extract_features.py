import argparse
import torch
from pathlib import Path
import h5py
import logging
from types import SimpleNamespace
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import time
import matplotlib.pyplot as plt

from . import extractors
from .utils.base_model import dynamic_load
from .utils.tools import map_tensor

'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 10240,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
}


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
    }

    def __init__(self, root, conf):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        for g in conf.globs:
            self.paths += list(Path(root).glob('**/' + g))
        if len(self.paths) == 0:
            raise ValueError(f'Could not find any image in root: {root}.')
        self.paths = sorted(list(set(self.paths)))
        self.paths = [i.relative_to(root) for i in self.paths]
        logging.info(f'Found {len(self.paths)} images in root {root}.')

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and max(w, h) > self.conf.resize_max:
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': path.as_posix(),
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.paths)


@torch.no_grad()
def main(conf, image_dir, export_dir, as_half=False):
    logging.info('Extracting local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    loader = ImageDataset(image_dir, conf['preprocessing'])
    loader = torch.utils.data.DataLoader(loader, num_workers=1)

    feature_path = Path(export_dir, conf['output'] + '_' + image_dir.stem + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    feature_file = h5py.File(str(feature_path), 'a')

    for data in tqdm(loader):
        pred = model(map_tensor(data, lambda x: x.to(device)))
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        grp = feature_file.create_group(data['name'][0])
        for k, v in pred.items():
            grp.create_dataset(k, data=v)

        del pred

    feature_file.close()
    logging.info('Finished exporting features.')


def query(img_processed, superpoint, device):
    logging.info('Extracting features...')
    features = {}

    for i, img in enumerate(tqdm(img_processed)):
        this_img = img[1]
        o_size = [this_img.shape[1], this_img.shape[0]]

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        this_img = transform(Image.fromarray(np.uint8(this_img))).unsqueeze(0)
        data = {
            'name': [img[0]],
            'image': this_img,
            'original_size': torch.tensor([o_size]),
        }

        pred = superpoint(map_tensor(data, lambda x: x.to(device)))
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        pred['image_size'] = original_size = data['original_size'][0].numpy()

        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

        features.update({img[0]: pred})
        time.sleep(0.001)

    return features
