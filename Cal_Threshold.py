import time
import os
import logging
import heapq
import matplotlib.pyplot as plt
import shutil
import h5py
from tqdm import tqdm
from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, preprocess
from hloc import localize_sfm, visualization, triangulation, colmap_rela_func
from hloc import Offline, Online
from netvlad import mobile_netvlad_sort2

from hloc.utils.database import COLMAPDatabase
from .utils.parsers import names_to_pair

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

group = 'Test'
group2 = 'test'

dataset = Path('datasets/' + group + '/')
imagesDB_o = dataset / 'db_o'
imagesQ_o = dataset / 'queries_o'
imagesDB = dataset / 'db'
imagesQ = dataset / 'queries'

outputs = Path('outputs/' + group + '/')
outputsR = outputs / 'results'
model = outputs / 'model'
query_pairs = outputs / 'pairs/pairs-query-mbnet-radius.txt'
db_pairs = outputs / 'pairs/pairs-db-mbnet-topk.txt'
results = outputsR / (group + '_hloc_superpoint+superglue_mbnet.txt')


def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def extract_matches(pairs_path, matches_path):
    with open(str(pairs_path), 'r') as f:
        pairs = [p.split(' ') for p in f.read().split('\n')]

    hfile = h5py.File(str(matches_path), 'r')

    match_quality = []
    match_pair = []
    logging.info('Extracting info from SuperGlue results')
    for name0, name1 in tqdm(pairs):
        pair0 = names_to_pair(name0, name1)
        if pair0 in hfile:
            pair = pair0
        else:
            continue

        matches = hfile[pair]['matches0'].__array__()
        valid = matches > -1
        scores = hfile[pair]['matching_scores0'].__array__()
        matches = matches[valid]

        proportion = float(matches.size) / float(scores.size) * 100

        name_temp = [name0, name1]
        match_pair.append(name_temp)
        match_quality.append(proportion)

    hfile.close()

    return match_pair, match_quality


imglist_qe = getFileList(imagesQ, [], 'jpg')
for i in range(imglist_qe.__len__()):
    imglist_qe[i] = imglist_qe[i].split('/')[-1]

# Offline.main(imagesDB, imagesDB_o, outputs, db_pairs, feature_conf, matcher_conf, model)

# outputsR.mkdir(exist_ok=True, parents=True)
# if os.listdir(outputsR):
#     shutil.rmtree(outputsR, ignore_errors=True)
#     os.mkdir(outputsR)

# colmap_rela_func.online_preprocess(imagesQ, imagesQ_o, outputsR / 'queries_intrinsics.txt')
#
# extract_features.main(feature_conf, imagesQ, outputsR)
#
# mobile_netvlad_sort2.main(imagesQ, 0.17, 5, outputs / 'pairs')
#
# match_features.main_query(matcher_conf, query_pairs, feature_conf['output'], outputs, outputsR)

match_pair, match_quality = extract_matches(query_pairs,
                                            outputs / f"{feature_conf['output']}_{matcher_conf['output']}_{db_pairs.stem}.h5")

match_first_img = []
for i in range(match_pair.__len__()):
    match_first_img.append(match_pair[i][0])

img_for_pair = []
img_quality = []
for img_qe in imglist_qe:
    img_for_pair.append([match_pair[i][1] for i, v in enumerate(match_first_img) if v == img_qe])
    img_quality.append([match_quality[i] for i, v in enumerate(match_first_img) if v == img_qe])

qe_match_img_list = []
for i in range(img_for_pair.__len__()):
    qe_match_img_list.append(img_for_pair[i][list(map(img_quality[i].index, heapq.nsmallest(1, img_quality[i])))[0]])

score = 0
tlen = imglist_qe.__len__()
logging.info('Calculating Mobile NetVLAD threshold')
for i in tqdm(range(imglist_qe.__len__())):
    db = qe_match_img_list[i]
    qe = imglist_qe[i]

    if db == qe:
        tlen = tlen - 1
        continue
    else:
        temp = score
        score = score + mobile_netvlad_sort2.threshold_test(outputs / 'pairs', qe, db)
    time.sleep(0.001)

print(score / tlen)
