import os
import cv2
import pandas as pd
import multiprocessing
from numpy.random import seed
seed(10)
from tqdm.auto import tqdm
import numpy as np
import pickle

from ..data.utils import get_train_df_ohe, get_public_df_ohe, get_masks_precomputed

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--public-data", action='store_true')

args = parser.parse_args()

PUBLIC_DATA_FLAG = args.public_data

PATH_TO_MASKS_ROOT = '../input/hpa_cell_mask_public/' if PUBLIC_DATA_FLAG else '../input/hpa_cell_mask/'
NEGS_PATH = '../input/all_negs_public.pkl' if PUBLIC_DATA_FLAG else '../input/all_negs.pkl'
OUTPUT_PATH = '../input/negs_with_staining_public.pkl' if PUBLIC_DATA_FLAG else '../input/negs_with_staining.pkl'

num_cores = multiprocessing.cpu_count()

main_df = get_public_df_ohe() if PUBLIC_DATA_FLAG else get_train_df_ohe()

with open(NEGS_PATH, 'rb') as f:
    all_negs = pickle.load(f)

img_id_2_no_staining_cells = dict()
for img_id_cell_i in all_negs:
    img_id, cell_i = img_id_cell_i.split('__')
    if img_id not in img_id_2_no_staining_cells:
        img_id_2_no_staining_cells[img_id] = {cell_i}
    else:
        img_id_2_no_staining_cells[img_id].add(cell_i)


def is_neg_with_staining(img_path):
    img_id = os.path.basename(img_path)
    if img_id not in img_id_2_no_staining_cells:
        return True
    masks = get_masks_precomputed([img_path], PATH_TO_MASKS_ROOT)[0]
    n_cells = masks.max()

    return n_cells > len(img_id_2_no_staining_cells[img_id])


negs_with_staining = []
for img_path in tqdm(main_df.loc[main_df['Negative'] == 1, 'img_base_path'].values):
    if is_neg_with_staining(img_path):
        negs_with_staining.append(img_path)

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(negs_with_staining, f)