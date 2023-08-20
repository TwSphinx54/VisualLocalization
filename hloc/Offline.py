import logging
import os

from hloc import extract_features, match_features, colmap_rela_func, mobile_netvlad_sort


def main(imagesDB, imagesDB_o, outputs, db_pairs, feature_conf, matcher_conf, model):
    outputs.mkdir(exist_ok=True, parents=True)
    if os.path.exists(outputs / 'database.db'):
        logging.warning("ERROR: database path already exists -- will not modify it.")
        exit()

    colmap_rela_func.write_image_info_2_db(imagesDB, imagesDB_o, outputs / 'database.db', outputs)

    extract_features.main(feature_conf, imagesDB, outputs)

    colmap_rela_func.import_features(
        outputs / 'image_ids.npy',
        outputs / 'database.db',
        outputs / f"{feature_conf['output']}_db.h5")

    mobile_netvlad_sort.generate_db_score_vector(imagesDB, outputs / 'pairs')
    mobile_netvlad_sort.generate_pairs_radius(0.9328985895831737, 20, outputs / 'pairs')

    match_features.main_db(matcher_conf, db_pairs, feature_conf['output'], outputs)

    colmap_rela_func.import_matches(
        outputs / 'image_ids.npy',
        outputs / 'database.db',
        db_pairs,
        outputs / f"{feature_conf['output']}_{matcher_conf['output']}_{db_pairs.stem}.h5",
        min_match_score=None,
        skip_geometric_verification=False)

    colmap_rela_func.geometric_verification(outputs / 'database.db', db_pairs, colmap_path='colmap')

    colmap_rela_func.run_triangulation(
        model,
        outputs / 'database.db',
        imagesDB,
        colmap_path='colmap')
