"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

import glob
from tqdm import tqdm
def make_grid_dataset(dir, test=False):
    images = []
    sN = glob.glob(dir + '/*')
    for sN_i in tqdm(sN):
        for sN_contents in glob.glob(sN_i + '/*'):
            if len(os.listdir(sN_contents)) < 75:
                    continue
            else:
                for fname in glob.glob(sN_contents + '/*.png'):
                    images.append(fname)
                if test:
                    break
                
    return images

