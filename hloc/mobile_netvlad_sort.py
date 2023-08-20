import cv2
import os
import logging
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from itertools import compress
import time
import heapq

from pathlib import Path
import torch


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


export_path = 'hloc/checkpoints/mobilenetvlad_depth-0.35'


def generate_db_score_vector(org_img_folder, results):
    # We start a session using a temporary fresh Graph
    org_img_folder_db = org_img_folder
    imglist_db = getFileList(org_img_folder_db, [], 'jpg')

    FeatureL_db = []
    ImgN_db = []
    logging.info('Generating database pictures\' score vector: ')
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], export_path)
        y = sess.graph.get_tensor_by_name('descriptor:0')
        x = sess.graph.get_tensor_by_name('image:0')
        for imgpath in tqdm(imglist_db):
            imgsubpath = imgpath.split('/')[-1]

            inim = cv2.imread(imgpath)
            inim = cv2.cvtColor(inim, cv2.COLOR_BGR2GRAY)
            inim = cv2.resize(inim, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            batch = np.expand_dims(inim, axis=0)
            batch = np.expand_dims(batch, axis=3)
            result = sess.run(y, feed_dict={x: batch})

            ##Dilated convolution
            tensor_result = torch.from_numpy(result)
            tensor_result = tensor_result.reshape([1, 64, 64])

            ##Average pooling
            # Averagepool_pool = torch.nn.AdaptiveAvgPool2d((16, 16))  # top5-9
            Averagepool_pool = torch.nn.AdaptiveAvgPool2d((30, 30))  # top20
            Averagepool_pool = Averagepool_pool.cuda()
            tensor_result = Averagepool_pool(tensor_result)
            # tensor_result = tensor_result.reshape([1, 256])
            tensor_result = tensor_result.reshape([1, 900])

            FeatureL_db.append(tensor_result.detach().numpy())
            ImgN_db.append(imgsubpath)

            time.sleep(0.01)

        Path(results).mkdir(exist_ok=True, parents=True)
        FeatureL_db = np.array(FeatureL_db)
        np.save(Path(results) / 'FeatureL_db_mbnet.npy', FeatureL_db)
        ImgN_db = np.array(ImgN_db)
        np.save(Path(results) / 'ImgN_db_mbnet.npy', ImgN_db)


def generate_pairs_radius(threshold_reasonable, threshold_max, results):
    FeatureL_db = np.load(Path(results) / 'FeatureL_db_mbnet.npy')
    FeatureL_db = FeatureL_db.tolist()
    ImgN_db = np.load(Path(results) / 'ImgN_db_mbnet.npy')
    ImgN_db = ImgN_db.tolist()

    logging.info('Sorting by picture diff: ')
    if os.path.exists(Path(results) / 'pairs-db-mbnet-radius.txt'):
        with open(Path(results) / 'pairs-db-mbnet-radius.txt', 'a+') as f:
            f.truncate(0)

    for i in tqdm(range(FeatureL_db.__len__())):
        diff = []
        for j in range(FeatureL_db.__len__()):
            tensor_j = torch.from_numpy(np.array(FeatureL_db[j]))
            tensor_i = torch.from_numpy(np.array(FeatureL_db[i]))

            out_diff = torch.norm(torch.abs(torch.sub(input=tensor_j, alpha=1, other=tensor_i)))
            diff.append(out_diff.cpu().detach().numpy())

        index = [j < threshold_reasonable for j in diff]
        diff_f = list(compress(diff, index))
        n_db_f = list(compress(ImgN_db, index))

        SimImg5 = list(map(diff_f.index, heapq.nsmallest(threshold_max, diff_f)))
        for smImgPath in SimImg5:
            with open(results / 'pairs-db-mbnet-radius.txt', 'a+') as f:
                if (smImgPath == SimImg5[-1]) & (i == FeatureL_db.__len__() - 1):
                    f.write(ImgN_db[i] + ' ' + n_db_f[smImgPath])
                else:
                    f.write(ImgN_db[i] + ' ' + n_db_f[smImgPath] + '\n')

        time.sleep(0.001)


def generate_pairs_topk(img_processed, FeatureL_db, ImgN_db, k, sess, y, x):
    FeatureL_qe = []
    ImgN_qe = []

    logging.info('Generating query pictures\' score vector: ')
    for this_img in tqdm(img_processed):
        inim = cv2.cvtColor(this_img[1], cv2.COLOR_BGR2GRAY)
        inim = cv2.resize(inim, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(inim, axis=0)
        batch = np.expand_dims(batch, axis=3)
        result = sess.run(y, feed_dict={x: batch})

        ##Dilated convolution
        tensor_result = torch.from_numpy(result)
        tensor_result = tensor_result.reshape([1, 64, 64])

        ##Average pooling
        # Averagepool_pool = torch.nn.AdaptiveAvgPool2d((16, 16))  # top5-9
        Averagepool_pool = torch.nn.AdaptiveAvgPool2d((30, 30))  # top20
        Averagepool_pool = Averagepool_pool.cuda()
        tensor_result = Averagepool_pool(tensor_result)
        # tensor_result = tensor_result.reshape([1, 256])
        tensor_result = tensor_result.reshape([1, 900])

        FeatureL_qe.append(tensor_result.detach().numpy())
        ImgN_qe.append(this_img[0])

    logging.info('Sorting by picture diff: ')
    pairs = []
    for i in tqdm(range(FeatureL_qe.__len__())):
        diff = []
        for j in range(FeatureL_db.__len__()):
            tensor_j = torch.from_numpy(np.array(FeatureL_db[j]))
            tensor_i = torch.from_numpy(np.array(FeatureL_qe[i]))

            out_diff = torch.norm(torch.abs(torch.sub(input=tensor_j, alpha=1, other=tensor_i)))
            diff.append(out_diff.cpu().detach().numpy())

        SimImg5 = list(map(diff.index, heapq.nsmallest(k, diff)))
        for smImgPath in SimImg5:
            pairs.append([ImgN_qe[i], ImgN_db[smImgPath]])

        time.sleep(0.001)

    return pairs


def generate_pairs_topk_withIO(org_img_folder, k, results):
    org_img_folder_qe = org_img_folder
    imglist_qe = getFileList(org_img_folder_qe, [], 'jpg')

    FeatureL_db = np.load(Path(results) / 'FeatureL_db_mbnet.npy')
    FeatureL_db = FeatureL_db.tolist()
    ImgN_db = np.load(Path(results) / 'ImgN_db_mbnet.npy')
    ImgN_db = ImgN_db.tolist()

    FeatureL_qe = []
    ImgN_qe = []

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], export_path)
        tf.get_default_graph()
        y = sess.graph.get_tensor_by_name('descriptor:0')
        x = sess.graph.get_tensor_by_name('image:0')

        logging.info('Generating query pictures\' score vector: ')
        for imgpath in tqdm(imglist_qe):
            imgsubpath = imgpath.split('/')[-1]

            inim = cv2.imread(imgpath)
            inim = cv2.cvtColor(inim, cv2.COLOR_BGR2GRAY)
            inim = cv2.resize(inim, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            batch = np.expand_dims(inim, axis=0)
            batch = np.expand_dims(batch, axis=3)
            result = sess.run(y, feed_dict={x: batch})

            ##Dilated convolution
            tensor_result = torch.from_numpy(result)
            tensor_result = tensor_result.reshape([1, 64, 64])

            ##Average pooling
            # Averagepool_pool = torch.nn.AdaptiveAvgPool2d((16, 16))  # top5-9
            Averagepool_pool = torch.nn.AdaptiveAvgPool2d((30, 30))  # top20
            Averagepool_pool = Averagepool_pool.cuda()
            tensor_result = Averagepool_pool(tensor_result)
            # tensor_result = tensor_result.reshape([1, 256])
            tensor_result = tensor_result.reshape([1, 900])

            FeatureL_qe.append(tensor_result.detach().numpy())
            ImgN_qe.append(imgsubpath)

    logging.info('Sorting by picture diff: ')
    if os.path.exists(Path(results) / 'pairs-query-mbnet-topk.txt'):
        with open(Path(results) / 'pairs-query-mbnet-topk.txt', 'a+') as f:
            f.truncate(0)

    for i in tqdm(range(FeatureL_qe.__len__())):
        diff = []
        for j in range(FeatureL_db.__len__()):
            tensor_j = torch.from_numpy(np.array(FeatureL_db[j]))
            tensor_i = torch.from_numpy(np.array(FeatureL_qe[i]))

            out_diff = torch.norm(torch.abs(torch.sub(input=tensor_j, alpha=1, other=tensor_i)))
            diff.append(out_diff.cpu().detach().numpy())

        SimImg5 = list(map(diff.index, heapq.nsmallest(k, diff)))
        for smImgPath in SimImg5:
            with open(Path(results) / 'pairs-query-mbnet-topk.txt', 'a+') as f:
                if (smImgPath == SimImg5[-1]) & (i == FeatureL_qe.__len__() - 1):
                    f.write(ImgN_qe[i] + ' ' + ImgN_db[smImgPath])
                else:
                    f.write(ImgN_qe[i] + ' ' + ImgN_db[smImgPath] + '\n')

        time.sleep(0.001)


def threshold_test(results, qe_name, db_name):
    FeatureL_db = np.load(results / 'FeatureL_db_mbnet.npy')
    FeatureL_db = FeatureL_db.tolist()
    ImgN_db = np.load(results / 'ImgN_db_mbnet.npy')
    ImgN_db = ImgN_db.tolist()

    index_qe = ImgN_db.index(qe_name)
    index_db = ImgN_db.index(db_name)

    feature_qe = np.array(FeatureL_db[index_qe])
    feature_db = np.array(FeatureL_db[index_db])

    return np.linalg.norm(feature_qe - feature_db)
