import os
import shutil

from hloc import extract_features, match_features, colmap_rela_func, pose_degensac, mobile_netvlad_sort, localize_sfm


def main(imagesQ, imagesQ_o, outputs, outputsR, query_pairs, feature_conf, matcher_conf, model):
    outputsR.mkdir(exist_ok=True, parents=True)
    if os.listdir(outputsR):
        shutil.rmtree(outputsR, ignore_errors=True)
        os.mkdir(outputsR)

    colmap_rela_func.online_preprocess(imagesQ, imagesQ_o, outputsR / 'queries_intrinsics.txt')

    extract_features.main(feature_conf, imagesQ, outputsR)

    mobile_netvlad_sort.generate_pairs_topk_withIO(imagesQ, 3, outputs / 'pairs')

    match_features.main_query(matcher_conf, query_pairs, feature_conf['output'], outputs, outputsR)

    # pose_degensac.degensac_getRT(outputsR, outputsR, query_pairs, outputsR / f"{feature_conf['output']}_queries.h5",
    #                              outputs / f"{feature_conf['output']}_db.h5",
    #                              outputsR / f"{feature_conf['output']}_{matcher_conf['output']}_{query_pairs.stem}.h5")

    localize_sfm.main(
        model,
        outputsR / 'queries_intrinsics.txt',
        query_pairs,
        outputsR / f"{feature_conf['output']}_queries.h5",
        outputsR / f"{feature_conf['output']}_{matcher_conf['output']}_{query_pairs.stem}.h5",
        outputsR / 'results.txt',
        ransac_thresh=250)
