import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os
import multiprocessing
import logging
from random import sample
import pandas as pd
import gc
from collections import Counter

from ..data.utils import get_train_df_ohe, get_public_df_ohe, get_class_names, get_masks_precomputed, open_rgb, get_cell_img_with_mask

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--precomputed-knn-graph-path", default=None)
parser.add_argument("--output-path", default=None)

args = parser.parse_args()
FOLD_I = args.fold
OUTPUT_PATH = args.output_path
if OUTPUT_PATH is None:
    OUTPUT_PATH = f'../output/mitotic_neighbours_fold_{FOLD_I}.pkl'

PRECOMPUTED_KNN_GRAPH = args.precomputed_knn_graph_path

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(
    f'Mitotic spindle neighbours: {os.path.basename(OUTPUT_PATH)}'
)

#############################
train_df = get_train_df_ohe(clean_from_duplicates=True)
img_paths_train = list(train_df['img_base_path'].values)
trn_basepath_2_ohe_vector = {img: vec for img, vec in zip(train_df['img_base_path'], train_df.iloc[:, 2:].values)}

public_hpa_df = get_public_df_ohe(clean_from_duplicates=True)
public_basepath_2_ohe_vector = {img_path: vec for img_path, vec in
                                zip(public_hpa_df['img_base_path'], public_hpa_df.iloc[:, 2:].values)}

trn_basepath_2_ohe_vector.update(public_basepath_2_ohe_vector)

# mappings ids
img_base_path_2_id = dict()
img_id_2_base_path = dict()
for img_base_path in trn_basepath_2_ohe_vector.keys():
    img_id = os.path.basename(img_base_path)
    img_base_path_2_id[img_base_path] = img_id
    img_id_2_base_path[img_id] = img_base_path

class_names = get_class_names() + ['Nothing there']

all_embs_df = pd.read_parquet('../output/densenet121_embs.parquet')

cherrypicked_mitotic_spindle = pd.read_csv('../input/mitotic_cells_selection.csv')
cherrypicked_mitotic_spindle_img_cell = set(cherrypicked_mitotic_spindle[['ID', 'cell_i']].apply(tuple, axis=1).values)

mitotic_spindle_class_i = class_names.index('Mitotic spindle')


# folds
with open('../input/denoisining_folds.pkl', 'rb') as f:
    fold_2_imgId_2_maskIndices = pickle.load(f)

# ## fold encodings, labels, ids
img_id_mask_global = []

logger.info('Gathering fold encodings and labels')
img_ids_with_embs = set(all_embs_df.index.get_level_values(0))

idx_2_mitotic_img_cell = dict()

known_mitotic_indices = set()
for img_id, cell_indices in tqdm(fold_2_imgId_2_maskIndices[FOLD_I].items(), desc='Gathering fold encodings and labels'):
    if img_id not in img_ids_with_embs: continue

    for cell_i in cell_indices:
        if (img_id, cell_i) in cherrypicked_mitotic_spindle_img_cell:
            known_mitotic_indices.add(len(img_id_mask_global))
        idx_2_mitotic_img_cell[len(img_id_mask_global)] = (img_id, cell_i)
        img_id_mask_global.append((img_id, cell_i))

del all_embs_df
gc.collect()

# KNN graph

with open(PRECOMPUTED_KNN_GRAPH, 'rb') as f:
    knn_graph_200 = pickle.load(f)
logger.info('Precomputed KNN graph loaded!')

neighbour_indices_all = Counter(knn_graph_200[list(known_mitotic_indices)].indices)

neighbour_img_cell = {idx_2_mitotic_img_cell[neighb_idx]: neighb_count for neighb_idx, neighb_count in neighbour_indices_all.items()
                      if neighb_count >= 3 and neighb_idx not in known_mitotic_indices}

logger.info(f'Discovered {len(neighbour_img_cell)} neighbours.')

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(neighbour_img_cell, f)