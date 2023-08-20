import matplotlib.pyplot as plt

from hloc import extract_features, match_features, colmap_rela_func, pose_degensac, mobile_netvlad_sort, localize_sfm


def main(imagesQ_o, outputs, feature_conf, matcher_conf, model):
    features_db, FeatureL_db, ImgN_db, superpoint, superglue, device, sess, y, x = \
        colmap_rela_func.preload(outputs / f"{feature_conf['output']}_db.h5",
                                 outputs / 'pairs/FeatureL_db_mbnet.npy',
                                 outputs / 'pairs/ImgN_db_mbnet.npy',
                                 feature_conf, matcher_conf)

    print('==========================================================================================================')
    input("Press Enter to continue...")

    intrinsics, qe_img_processed = colmap_rela_func.new_online_preprocess(imagesQ_o)

    pairs = mobile_netvlad_sort.generate_pairs_topk(qe_img_processed, FeatureL_db, ImgN_db, 3, sess, y, x)

    features_qe = extract_features.query(qe_img_processed, superpoint, device)

    # this_img = qe_img_processed[2]
    # plt.rcParams['savefig.dpi'] = 300
    # plt.rcParams['figure.figsize'] = (36.0, 64.0)
    # plt.imshow(this_img[1])
    # kps = features_qe[this_img[0]]['keypoints']
    # plt.plot(kps[:, 0], kps[:, 1], 'ro')
    # plt.savefig(this_img[0])
    # exit()

    matches = match_features.main_query(pairs, features_qe, features_db, superglue, device)

    # pose_degensac.degensac_getRT(intrinsics, outputs, pairs, features_qe, features_db, matches)

    localize_sfm.main(model, intrinsics, pairs, features_qe, matches, outputs / 'test.txt')

    # colmap_rela_func.online_postprocess(imagesQ, imagesQ_o, imagesQ_old)
