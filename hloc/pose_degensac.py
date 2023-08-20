import numpy as np
import cv2
import pydegensac
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import h5py
import logging
import time
from scipy.spatial.transform import Rotation as Rot

from .utils.parsers import names_to_pair


def degensac_getRT(intrinsics, result, pairs, features_qe, features_db, match):
    logging.info('Estimating Pose:')

    i_qe = 0
    res = []
    inl = 0
    temp = []
    prev_qe = pairs[0][0]
    prev = pairs[0][0]

    for this_pair in tqdm(pairs):
        if this_pair[0] != prev_qe:
            i_qe += 1
            prev_qe = this_pair[0]

        keypoints1 = features_qe[this_pair[0]]['keypoints']
        keypoints1 += 0.5  # COLMAP origin
        keypoints2 = features_db[this_pair[-1]]['keypoints']
        keypoints2 += 0.5  # COLMAP origin
        kps1 = np.asarray(keypoints1, np.float32)
        kps2 = np.asarray(keypoints2, np.float32)

        pair0 = names_to_pair(this_pair[0], this_pair[-1])
        if pair0 in match:
            pair = pair0
        else:
            continue

        matches = match[pair]['matches0'].__array__()
        valid = matches > -1
        matches = np.stack([np.where(valid)[0], matches[valid]], -1)

        n_kps1 = []
        n_kps2 = []
        for this_match in matches:
            n_kps1.append(kps1[this_match[0]])
            n_kps2.append(kps2[this_match[-1]])
        n_kps1 = np.asarray(n_kps1, np.float32)
        n_kps2 = np.asarray(n_kps2, np.float32)

        ##get R, T from Fundamental matrix
        th = 0.5
        n_iter = 50000
        cmp_F, cmp_mask = pydegensac.findFundamentalMatrix(n_kps1, n_kps2, th, 0.999, n_iter,
                                                           enable_degeneracy_check=True)
        # Essential matrix
        K = np.array(intrinsics[i_qe])
        E = np.transpose(K) @ cmp_F @ K
        # recover pose
        R = np.zeros((3, 3))
        T = np.zeros((3, 1))
        cv2.recoverPose(E, n_kps1, n_kps2, K, R, T)
        R = Rot.from_matrix(R)
        R = R.as_quat()

        inlier = int(deepcopy(cmp_mask).astype(np.float32).sum())
        R = [str(tt) for tt in R]
        T = [str(tt) for tt in T.transpose()[0]]

        if this_pair == pairs[0]:
            temp = [str(this_pair[0]), str(this_pair[-1])] + R + T
            inl = inlier
        elif (this_pair[0] == prev) & (inlier > inl):
            temp = [str(this_pair[0]), str(this_pair[-1])] + R + T
            inl = inlier
        elif this_pair[0] != prev:
            res.append(temp)
            temp = [str(this_pair[0]), str(this_pair[-1])] + R + T
            inl = inlier

        prev = this_pair[0]

    with open(result / 'result.txt', 'w+') as f:
        for k in res:
            for i in k:
                f.write(i + ' ')
            f.write('\n')

    logging.info('Done!')
