import re
import time
import os
import logging
import exifread
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pformat
import h5py
import pickle
# import pycolmap
from collections import defaultdict
import cv2 as cv
# from sklearn.neighbors import KDTree

from hloc import extract_features, match_features, preprocess
from hloc import localize_sfm, colmap_rela_func
from hloc import Offline, Online
# from netvlad import netvlad_sort, mobile_netvlad_sort2
from hloc import mobile_netvlad_sort

from hloc.utils.database import COLMAPDatabase

from hloc.utils.read_write_model import read_model
from hloc.utils.parsers import (parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

group = 'Square'
group2 = 'square'

dataset = Path('datasets/' + group + '/')
imagesDB_o = dataset / 'db_o'
imagesQ_o = dataset / 'queries_o'
imagesQ_old = dataset / 'queries_old'
imagesDB = dataset / 'db'
imagesQ = dataset / 'queries'

outputs = Path('outputs/' + group + '/')
outputsR = outputs / 'results'
model = outputs / 'model'
query_pairs = outputs / 'pairs/pairs-query-mbnet-radius.txt'
db_pairs = outputs / 'pairs/pairs-db-mbnet-topk.txt'
results = outputsR / (group + '_hloc_superpoint+superglue_mbnet.txt')

mobile_netvlad_sort.generate_db_score_vector(imagesDB, outputs / 'pairs')

# extract_features.main(feature_conf, imagesDB, outputs)

# # pts = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
# pts = np.random.random((500, 2))
# print(pts)
# new_pt = np.array([[1.1, 2.1]])
# print(new_pt)
#
# tree = KDTree(pts)
# dist, ind = tree.query(new_pt, k=1)
# print(dist)
# print(ind)
# print(pts[ind[0][0]])

# t = np.array([1, 2, 3, 4, 5])
# s = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
#
# print(t)
# print(type(t))
# print(t.shape)
#
# print(s[1])
# print(type(s[1]))
# print(s[1].shape)
#
# t = np.append(t, s[1])
# print(t)

# fp = '/home/px/MyVL/datasets/SSSquare/db_o/00000000.jpg'
# img = cv.imread(fp)
#
# h = img.shape[0]
# w = img.shape[1]
#
# size = (720, 960)
# step = (200, 200)
#
# h_b = (h - size[0]) // step[0]
# w_b = (w - size[1]) // step[1]
#
# for h_id in range(h_b + 2):
#     if h_id == h_b + 1:
#         h_s = h - size[0]
#         h_e = h
#     else:
#         h_s = h_id * step[0]
#         h_e = size[0] + h_id * step[0]
#     for w_id in range(w_b + 2):
#         if w_id == w_b + 1:
#             w_s = w - size[1]
#             w_e = w
#         else:
#             w_s = w_id * step[1]
#             w_e = size[1] + w_id * step[1]
#
#         this_img = img[h_s:h_e, w_s:w_e]
#         cv.imwrite('/home/px/MyVL/outputs/Test/' + str(h_id) + '_' + str(w_id) + '.jpg', this_img)
