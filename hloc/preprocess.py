# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:50:32 2021

@author: LENOVO
"""

import cv2
import os
from tqdm import tqdm
import argparse
import re
import exifread
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--path_in', type=str, default='', help='path of input images')
parser.add_argument('--path_out', type=str, default='', help='path of output images')
parser.add_argument('--height', type=int, default=0, help='height of output images')
parser.add_argument('--width', type=int, default=0, help='width of ouput images')
parser.add_argument('--output_exif', type=str, default='', help='path of output exif messages')


def read_exif(img_name, rate):
    with open(img_name, 'rb') as f:
        tags = exifread.process_file(f)
    f.close()
    width = tags.get('Image ImageWidth', '0').values[0] * rate
    height = tags.get('Image ImageLength', '0').values[0] * rate
    # f35 = tags.get('EXIF FocalLengthIn35mmFilm', '0').values[0]
    # fl = width * f35 / 36
    fl = max(width, height) * 1.2
    cx = width / 2
    cy = height / 2

    return width, height, fl, cx, cy


def rename_images(datadir):
    fileList = os.listdir(datadir)
    os.chdir(datadir)
    num = 0

    logging.info('Renaming database images')
    for fileName in tqdm(fileList):
        pat = ".+\.(jpg|jpeg|JPG)"
        pattern = re.findall(pat, fileName)
        print('pattern[0]:', pattern)
        print('num：', num, 'filename:', fileName)
        name = "{:0>8d}".format(num)
        os.rename(fileName, (name + '.' + pattern[0]))
        num = num + 1


def resize_images(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    h, w = img.shape[0:2]
    if h > 1024 or w > 1024:
        rate = min(float(1024.0 / h), float(1024.0 / w))
    else:
        rate = 1
    h_ = int(h * rate)
    w_ = int(w * rate)
    img = cv2.resize(img, (w_, h_))

    return rate, img


if __name__ == '__main__':
    path_in = parser.parse_args().path_in
    path_out = parser.parse_args().path_out
    height = parser.parse_args().height
    width = parser.parse_args().width
    output_exif = parser.parse_args().output_exif

    # rename_images(path_in)

    # resize_images(path_in, path_out, height, width)

    # read_exif(path_in, output_exif)
