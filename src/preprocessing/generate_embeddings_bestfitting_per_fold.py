import os
import cv2
import pickle
import multiprocessing
from albumentations import VerticalFlip, HorizontalFlip, Rotate, GridDistortion, BboxParams
from numpy.random import seed
seed(10)
from tqdm.auto import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import warnings
warnings.simplefilter("ignore")
import sys
sys.path.insert(0, '../HPA-competition-solutions/bestfitting/src')
from networks.densenet import DensenetClass

from ..models.encodings_pretrained import BestfittingEncodingsModel
from ..data.datasets import ProteinMLDatasetModified
from ..data.utils import get_public_df_ohe, get_train_df_ohe, get_masks_precomputed

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--fold-part", type=int, default=1)
parser.add_argument("--public-data", action='store_true')

args = parser.parse_args()

IMG_HEIGHT = IMG_WIDTH = 1024
BATCH_SIZE = 8
NUM_CORES = multiprocessing.cpu_count()
PUBLIC_DATA_FLAG = args.public_data
PATH_TO_MASKS_ROOT = '../input/hpa_cell_mask_public/' if PUBLIC_DATA_FLAG else '../input/hpa_cell_mask/'
OUTPUT_PATH_EMBS = '../input/bestfitting_densenet_embs_public' if PUBLIC_DATA_FLAG else '../input/bestfitting_densenet_embs_train'
INFERENCE_STEP = 5
FOLD_I = args.fold
FOLD_PART = args.fold_part
IMGS_FOLDER = '../input/publichpa_1024' if PUBLIC_DATA_FLAG else '../input/hpa-single-cell-image-classification/train'

main_df = get_public_df_ohe() if PUBLIC_DATA_FLAG else get_train_df_ohe()

final_densenet_121 = torch.load('../input/pretrained_models/models/external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds/fold0/final.pth')
model = DensenetClass(in_channels=4, dropout=True, large=True)
model.load_state_dict(final_densenet_121['state_dict'])
bestfitting_features_model = BestfittingEncodingsModel(model)
bestfitting_features_model.cuda()
bestfitting_features_model.eval()


if not os.path.exists(OUTPUT_PATH_EMBS):
    os.makedirs(OUTPUT_PATH_EMBS)


with open('../input/denoisining_folds.pkl', 'rb') as f:
    fold_2_imgId_2_maskIndices = pickle.load(f)


vert_flip = VerticalFlip(always_apply=True)
hor_flip = HorizontalFlip(always_apply=True)
rot = Rotate(always_apply=True, limit=(89, 91))

    
def get_cell_embedding(img_ids, bestfitting_features_model=bestfitting_features_model, folder=IMGS_FOLDER,
                       img_id_2_mask_indices=None, classifier_img_height=IMG_HEIGHT, classifier_img_width=IMG_WIDTH,
                       batch_size=BATCH_SIZE, num_workers=NUM_CORES):

    dataset = ProteinMLDatasetModified(img_ids, folder=folder)
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True, collate_fn=lambda x: x
    )
    def get_cell_only(cell_bool_mask, img, background_val=0):
        cell_img = img.copy()
        cell_img[np.logical_not(cell_bool_mask)] = background_val
        images = [cell_img]

        flipped = vert_flip(image=cell_img)
        cell_img_flipped = flipped['image']   
        images.append(cell_img_flipped)
        
        flipped_2 = hor_flip(image=cell_img)
        cell_img_flipped_2 = flipped_2['image']
        images.append(cell_img_flipped_2)
            
        rot_aug = rot(image=cell_img)
        cell_img_rot = rot_aug['image']
        images.append(cell_img_rot)

        return np.stack(images)

    batch_i = -1
    for images_batch in loader:
        batch_i += 1
        images_batch = images_batch[:len(img_ids) - batch_i*batch_size]
        img_batch_ids = img_ids[batch_i*batch_size:(batch_i + 1)*batch_size]
        try:
            masks_batch = get_masks_precomputed(img_batch_ids, PATH_TO_MASKS_ROOT)
        except ValueError:
            continue
        
        for mask_i, mask_init in enumerate(masks_batch):
            mask_classification = cv2.resize(mask_init, (classifier_img_height, classifier_img_width))
            n_cells = mask_classification.max()
            
            img_current = images_batch[mask_i]
            img_id = img_batch_ids[mask_i]
            for cell_i in (range(1, n_cells + 1) if img_id_2_mask_indices is None else img_id_2_mask_indices[img_id]):
                if not os.path.exists(f'{OUTPUT_PATH_EMBS}/{img_id}__{cell_i}.npz'):
                    cell_mask_bool = mask_classification == cell_i
                    images = get_cell_only(cell_mask_bool, img_current, 
                                                    background_val=0)
                    images = images.astype(np.float32)
                    images = images.transpose((0, 3, 1, 2))
                    images = torch.from_numpy(images).cuda()
                    with torch.no_grad():
                        embeddings = bestfitting_features_model(images).mean(axis=0).detach().cpu().numpy()
                        np.savez_compressed(f'{OUTPUT_PATH_EMBS}/{img_id}__{cell_i}', embeddings)


img_ids = set(main_df['ID'].values)

fold_i_ids = [img_id for img_id in fold_2_imgId_2_maskIndices[FOLD_I].keys() if img_id in img_ids]


def all_cell_embs_present(img_id):
    mask = get_masks_precomputed([img_id], PATH_TO_MASKS_ROOT)[0]
    n_cells = mask.max()
    for cell_i in range(1, n_cells + 1):
        if not os.path.exists(f'{OUTPUT_PATH_EMBS}/{img_id}__{cell_i}.npz'):
            return False
    return True


fold_i_ids = [img_id for img_id in fold_i_ids if not all_cell_embs_present(img_id)]

for next_start_block_i in tqdm(range(FOLD_PART*len(fold_i_ids) // 6,
                                     (FOLD_PART + 1) * len(fold_i_ids) // 6 if FOLD_PART < 5 else len(fold_i_ids),
                                     INFERENCE_STEP)):
    get_cell_embedding(fold_i_ids[next_start_block_i: next_start_block_i + INFERENCE_STEP],
                       bestfitting_features_model,
                       img_id_2_mask_indices=fold_2_imgId_2_maskIndices[FOLD_I])

