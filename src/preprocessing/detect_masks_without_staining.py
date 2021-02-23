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
from ..data.datasets import DataGeneneratorRGB


IMG_HEIGHT = IMG_WIDTH = 512
BATCH_SIZE = 8
DETECTION_THRESHOLD = 150

PUBLIC_DATA_FLAG = True
PATH_TO_MASKS_ROOT = '../input/hpa_cell_mask_public/' if PUBLIC_DATA_FLAG else '../input/hpa_cell_mask/'
OUTPUT_PATH = '../input/all_negs_public.pkl' if PUBLIC_DATA_FLAG else '../input/all_negs.pkl'

num_cores = multiprocessing.cpu_count()

main_df = get_public_df_ohe() if PUBLIC_DATA_FLAG else get_train_df_ohe()


def compute_mask_green_vals(img_base_path, masks=None, return_green_means=True):
    green_img = cv2.imread(f'{img_base_path}_green.png', cv2.IMREAD_GRAYSCALE)
    green_vals = []
    if masks is not None and return_green_means:      
        for mask_i in range(1, masks[0].max() + 1):        
            cell_mask_bool = masks[0] == mask_i
            cell_green_val = np.max(green_img[cell_mask_bool])
            green_vals.append(cell_green_val)

    return green_vals
            

if DETECTION_THRESHOLD is None:
    indices = [0, 4, 9, 14, 15, 18, 22, 27, 29, 30, 31]
    green_vals = []
    for i, img_path in enumerate(main_df.loc[main_df['Negative'] == 1, 'img_base_path'].values):
        if i not in indices:
            masks = get_masks_precomputed([os.path.basename(img_path)], PATH_TO_MASKS_ROOT)
            green_vals.extend(compute_mask_green_vals(img_path, masks=masks))
    DETECTION_THRESHOLD = np.quantile(green_vals, 0.7)


def get_neg_cells(img_paths, img_height=2048, img_width=2048,
                  batch_size=BATCH_SIZE, detection_threshold=DETECTION_THRESHOLD):

    data_gen = DataGeneneratorRGB(img_paths,
                               shuffle=False, batch_size=batch_size,
                               resized_height=img_height, resized_width=img_width)
    
    results = []
    for batch_i in range(len(img_paths)//batch_size + (1 if len(img_paths)%batch_size != 0 else 0)):
        
        images_batch = data_gen.__getitem__(batch_i)
        images_batch = images_batch[:len(img_paths) - batch_i*batch_size, :, :]
        img_batch_paths = img_paths[batch_i*batch_size:(batch_i + 1)*batch_size]
        try:
            masks_batch = get_masks_precomputed(img_batch_paths, PATH_TO_MASKS_ROOT)
        except ValueError:
            continue
        
        for mask_i, mask_init in enumerate(masks_batch):
            mask = cv2.resize(mask_init, (img_height, img_width))
            n_cells = mask.max()

            green_img = images_batch[mask_i][:, :, 1]
            for cell_i in range(1, n_cells + 1):
                cell_mask_bool = mask == cell_i
                cell_green_val = np.max(green_img[cell_mask_bool])
            
                if cell_green_val <= detection_threshold:
                    results.append(f'{os.path.basename(img_batch_paths[mask_i])}__{cell_i}')
    return results


all_negs = []
inference_step = BATCH_SIZE
for next_start_block_i in tqdm(range(0, main_df.shape[0], inference_step),
                               desc=('Detecting cells without staining (rule-based) in '
                                       f'{"public" if PUBLIC_DATA_FLAG else "train"} data')):
    all_negs.extend(get_neg_cells(main_df['img_base_path'].values[next_start_block_i: next_start_block_i + inference_step]))

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(all_negs, f)
