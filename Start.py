from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, Offline, Online_withIO

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
group = 'Square'

dataset = Path('datasets/' + group + '/')
imagesDB_o = dataset / 'db_o'
imagesQ_o = dataset / 'queries_o'
imagesDB = dataset / 'db'
imagesQ = dataset / 'queries'

outputs = Path('outputs/' + group + '/')
outputsR = outputs / 'results'
model = outputs / 'model'
query_pairs = outputs / 'pairs/pairs-query-mbnet-topk.txt'
db_pairs = outputs / 'pairs/pairs-db-mbnet-radius.txt'

print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

Offline.main(imagesDB, imagesDB_o, outputs, db_pairs, feature_conf, matcher_conf, model)
Online_withIO.main(imagesQ, imagesQ_o, outputs, outputsR, query_pairs, feature_conf, matcher_conf, model / '0')
