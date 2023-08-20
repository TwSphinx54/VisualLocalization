import logging
import time

import cv2
from tqdm import tqdm
import h5py
import numpy as np
import subprocess
import torch
import tensorflow.compat.v1 as tf

import os
import shutil

from .utils.read_write_model import CAMERA_MODEL_NAMES
from .utils.database import COLMAPDatabase
from .utils.parsers import names_to_pair
from hloc import preprocess
from . import extractors
from . import matchers
from .utils.base_model import dynamic_load

export_path = 'hloc/checkpoints/mobilenetvlad_depth-0.35'


def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        # path = os.listdir(dir)
        # path.sort(key=lambda x: int(x[:-(ext.__len__() + 1)]))
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def online_preprocess(query_path, query_path_o, output):
    query_path.mkdir(exist_ok=True, parents=True)
    imglist = getFileList(query_path_o, [], 'jpg')

    logging.info('Generating query intrinsics')
    ws = ' '
    for img_name in tqdm(imglist):
        rate, img = preprocess.resize_images(img_name)
        width, height, fl, cx, cy = preprocess.read_exif(img_name, rate)
        cv2.imwrite(str(query_path) + '/' + img_name.split('/')[-1], img)

        k = str(0.)

        imgname = img_name[str(query_path_o).__len__() + 1:]

        content = (
            imgname, 'SIMPLE_RADIAL', str(int(width)), str(int(height)), str(fl), str(cx), str(cy),
            k)
        with open(output, 'a+') as f:
            if img_name == imglist[-1]:
                f.write(ws.join(content))
            else:
                f.write(ws.join(content) + '\n')

        time.sleep(0.001)


def write_image_info_2_db(image_dir, image_dir_o, database_path, output):
    # preprocess.rename_images(image_dir)

    image_dir.mkdir(exist_ok=True, parents=True)
    imglist = getFileList(image_dir_o, [], 'jpg')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    i = 1
    imgid = {}

    logging.info('Writing images information to database:')
    for img_name in tqdm(imglist):
        rate, img = preprocess.resize_images(img_name)
        width, height, fl, cx, cy = preprocess.read_exif(img_name, rate)
        cv2.imwrite(str(image_dir) + '/' + img_name.split('/')[-1], img)

        k = 0.

        imgname = img_name[str(image_dir_o).__len__() + 1:]
        db.add_image(imgname, camera_id=i, image_id=i)
        params = np.array((fl, cx, cy, k))
        db.add_camera(CAMERA_MODEL_NAMES['SIMPLE_RADIAL'].model_id, width, height, params, camera_id=i)

        imgid[imgname] = i
        i = i + 1

        time.sleep(0.001)

    db.commit()
    db.close()

    np.save(output / 'image_ids.npy', imgid)


def import_features(image_ids_path, database_path, features_path):
    image_ids = np.load(image_ids_path, allow_pickle=True).item()

    logging.info('Importing features into the database...')
    hfile = h5py.File(str(features_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = hfile[image_name]['keypoints'].__array__()
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    hfile.close()
    db.commit()
    db.close()


def import_matches(image_ids_path, database_path, pairs_path, matches_path, min_match_score=None,
                   skip_geometric_verification=False):
    image_ids = np.load(image_ids_path, allow_pickle=True).item()

    logging.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split(' ') for p in f.read().split('\n')]

    hfile = h5py.File(str(matches_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        pair0 = names_to_pair(name0, name1)
        pair1 = names_to_pair(name0, name1)
        if pair0 in hfile:
            pair = pair0
        elif pair1 in hfile:
            pair = pair1
        else:
            raise ValueError(f'Could not find pair {(name0, name1)}')

        matches = hfile[pair]['matches0'].__array__()
        valid = matches > -1
        if min_match_score:
            scores = hfile[pair]['matching_scores0'].__array__()
            valid = valid & (scores > min_match_score)
        matches = np.stack([np.where(valid)[0], matches[valid]], -1)

        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    hfile.close()
    db.commit()
    db.close()


def geometric_verification(database_path, pairs_path, colmap_path):
    logging.info('Performing geometric verification of the matches...')
    cmd = [
        str(colmap_path), 'matches_importer',
        '--database_path', str(database_path),
        '--match_list_path', str(pairs_path),
        '--match_type', 'pairs']
    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with matches_importer, exiting.')
        exit(ret)


def run_triangulation(model_path, database_path, image_dir, colmap_path):
    logging.info('Running the triangulation...')
    model_path.mkdir(exist_ok=True, parents=True)

    cmd = [
        str(colmap_path), 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(model_path),
        '--Mapper.ba_refine_focal_length', '1',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0']
    logging.info(' '.join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with mapper, exiting.')
        exit(ret)

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer', '--path', str(model_path / '0')])
    stats_raw = stats_raw.decode().split("\n")
    stats = dict()
    for stat in stats_raw:
        if stat.startswith("Registered images"):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith("Points"):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])

    return stats
