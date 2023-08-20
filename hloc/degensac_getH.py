import numpy as np
import cv2
import pydegensac
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import os


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


def verify_pydegensac(kps1, kps2, tentatives, th=4.0, n_iter=2000):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
    H, mask = pydegensac.findHomography(src_pts, dst_pts, th, 0.99, n_iter)
    print('found {} inliers'.format(int(deepcopy(mask).astype(np.float32).sum())))
    return H, mask


def degensac_getRT(intripath, result, netvlad_pairs, k, query_path, db_path):
    intrinsics = np.load(Path(intripath) / 'intrinsics.npy')
    intrinsics = intrinsics.tolist()

    imglist = getFileList(query_path, [], 'jpg')
    with open(netvlad_pairs, 'r') as f:
        for img_name in tqdm(imglist):
            for z in range(k):
                pair = f.readline()
                refer = pair.split(' ')[1].split('\\')[1].split('.')[0]

                img1_pth = img_name
                img2_pth = db_path + '/' + refer + '.jpg'
                img1 = cv2.cvtColor(cv2.imread(img1_pth), cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(cv2.imread(img2_pth), cv2.COLOR_BGR2RGB)
                # SIFT is not available by pip install, so lets use AKAZE features
                det = cv2.AKAZE_create(descriptor_type=3, threshold=0.00001)
                kps1, descs1 = det.detectAndCompute(img1, None)
                kps2, descs2 = det.detectAndCompute(img2, None)
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(descs1, descs2, k=2)
                matchesMask = [False for i in range(len(matches))]
                # SNN ratio test
                for i, (m, n) in enumerate(matches):
                    if m.distance < 0.9 * n.distance:
                        matchesMask[i] = True
                tentatives = [m[0] for i, m in enumerate(matches) if matchesMask[i]]

                # get H matrix
                th = 4.0
                n_iter = 2000
                cmp_H, cmp_mask = verify_pydegensac(kps1, kps2, tentatives, th, n_iter)
                # get R, T from H
                K = np.array(intrinsics[z])
                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(cmp_H, K)

                with open(Path(result) / 'rt.txt', 'a+') as f1:
                    f1.write(img_name.split('\\')[-1] + '-' + str(refer) + ' ' + str(num) + '\n')
                    for y in range(num):
                        f1.write('\n')
                        f1.write(str(Rs[y]))
                        f1.write('\n')
                        f1.write(str(Ts[y]))
                        f1.write('\n')
                    f1.write('\n----------------------------\n')


if __name__ == '__main__':
    # test
    degensac_getRT('D:/Code/VLocalization/SSvl/output',
                   'D:/Code/VLocalization/SSvl/output',
                   'D:/Code/VLocalization/SSvl/output/pair/pairs-query-netvlad-dynamic.txt',
                   5, 'E:/Second2/Vloc/datasets/Outside/SSquare/images/queries',
                   'E:/Second2/Vloc/datasets/Outside/SSquare/images/db')
