import pickle
import numpy as np
import os
from sklearn.model_selection import KFold
from random import choice
from tqdm.auto import tqdm

from ..data.utils import get_public_df_ohe, get_train_df_ohe

N_FOLDS = 5

train_df = get_train_df_ohe()
img_paths_train = list(train_df['img_base_path'].values)
basepath_2_ohe_vector = {img:vec for img, vec in zip(train_df['img_base_path'], train_df.iloc[:, 3:-1].values)}


public_hpa_df_17 = get_public_df_ohe()
public_basepath_2_ohe_vector = {img_path:vec for img_path, vec in zip(public_hpa_df_17['img_base_path'],
                                                            public_hpa_df_17.iloc[:, 4:-1].values)}


basepath_2_ohe_vector.update(public_basepath_2_ohe_vector)


# mappings ids
img_base_path_2_id = dict()
img_id_2_base_path = dict()
for img_base_path in basepath_2_ohe_vector.keys():
    img_id = os.path.basename(img_base_path)
    img_base_path_2_id[img_base_path] = img_id
    img_id_2_base_path[img_id] = img_base_path


# ID 2 mask indices
with open('../input/all_negs.pkl', 'rb') as f:
    all_negs_trn = set(pickle.load(f))
with open('../input/all_negs_public.pkl', 'rb') as f:
    all_negs_public = set(pickle.load(f))


def get_id_2_masks(precomputed_mask_path):
    available_files = os.listdir(precomputed_mask_path)
    id_2_mask_indices = dict()
    for file_name in tqdm(available_files, desc=f'Generating id_2_masks mapping for {precomputed_mask_path}'):
        img_id = os.path.splitext(file_name)[0]
        masks_all = np.load(os.path.join(precomputed_mask_path, file_name))['arr_0']
        id_2_mask_indices[img_id] = list(range(1, masks_all.max() + 1))
    return id_2_mask_indices


id_2_mask_indices = get_id_2_masks('../input/hpa_cell_mask')
id_2_mask_indices_public = get_id_2_masks('../input/hpa_cell_mask_public')
id_2_mask_indices.update(id_2_mask_indices_public)


# Splitting masks into folds
fold_2_imgId_2_maskIndices = [dict() for _ in range(N_FOLDS)]
for img_id, mask_indices in tqdm(id_2_mask_indices.items(), desc='Splitting into folds..'):
    if len(mask_indices) == 1:
        fold_i = choice([4]*10 + [3]*7 + [2]*5 + [1, 0])
        fold_2_imgId_2_maskIndices[fold_i][img_id] = mask_indices
        continue
    kf = KFold(n_splits=min(N_FOLDS, len(mask_indices)), shuffle=True, random_state=41)
    fold_i = 0
    for _, fold_indices in kf.split(range(len(mask_indices))):
        mask_idx_fold = [mask_indices[i] for i in fold_indices]
        if img_id in fold_2_imgId_2_maskIndices[fold_i]:
            fold_2_imgId_2_maskIndices[fold_i][img_id].extend(mask_idx_fold)
        else:
            fold_2_imgId_2_maskIndices[fold_i][img_id] = mask_idx_fold
        fold_i += 1


# fold sizes
for fold_i in range(N_FOLDS):
    size = 0
    for _, masks in fold_2_imgId_2_maskIndices[fold_i].items():
        size += len(masks)
    print(f'Fold {fold_i}: {size}')

with open('../input/denoisining_folds.pkl', 'wb') as f:
    pickle.dump(fold_2_imgId_2_maskIndices, f)