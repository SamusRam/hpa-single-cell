import sys
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
import imagehash
from tqdm import tqdm
import pickle
import mlcrate as mlc

from ..data.utils import get_public_df_ohe, get_train_df_ohe


def imread_custom(img_basepath, color):
    img = Image.open(f'{img_basepath}_{color}.png')
    return img


# https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72534
def generate_hash(df, colors, dataset='train', imread_func=imread_custom, is_update=False):
    cache_fname = f'output/{dataset}_hash_maps.pkl'

    hash_maps = {}
    for color in colors:
        hash_maps[color] = []
        for img_basepath in tqdm(df['img_base_path'].values, desc=f'{dataset}_{color}'):
            img = imread_func(img_basepath, color)
            hash = imagehash.phash(img)
            hash_maps[color].append(hash)

    with open(cache_fname, 'wb') as dbfile:
        pickle.dump(hash_maps, dbfile)

    for color in colors:
        df[color] = hash_maps[color]

    return df

def calc_hash(params):
    color, th, base_external_hash, base_train_hash, train_ids, external_ids = params

    external_hash = base_external_hash.reshape(1, -1)  # 1*m

    train_idxes_list = []
    external_idxes_list = []
    hash_list = []

    step = 5
    for train_idx in tqdm(range(0, len(base_train_hash), step), desc=color):
        train_hash = base_train_hash[train_idx:train_idx + step].reshape(-1, 1)  # n*1
        hash = train_hash - external_hash  # n*m
        train_idxes, external_idxes = np.where(hash <= th)
        hash = hash[train_idxes, external_idxes]

        train_idxes = train_idxes + train_idx

        train_idxes_list.extend(train_idxes.tolist())
        external_idxes_list.extend(external_idxes.tolist())
        hash_list.extend(hash.tolist())

    df = pd.DataFrame({
        'Train': train_ids[train_idxes_list],
        'Extra': external_ids[external_idxes_list],
        'Sim%s' % color[:1].upper(): hash_list
    })
    return df

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    threshold = 12

    # train set images
    train_df = get_train_df_ohe()

    # external images
    public_df = get_public_df_ohe()

    colors = ['red', 'green', 'blue']
    train_df = generate_hash(train_df, colors,
                           dataset='train', is_update=False)
    public_df = generate_hash(public_df, colors,
                                  dataset='external', is_update=False)

    pool = mlc.SuperPool(3)
    params = []
    for color in colors:
        base_tran_hash = train_df[color].values
        base_external_hash = public_df[color].values

        train_ids = train_df['img_base_path'].values
        external_ids = public_df['img_base_path'].values

        base_hash_all = np.concatenate((base_tran_hash, base_external_hash))
        ids_all = np.concatenate((train_ids, external_ids))

        params.append((color, threshold, base_hash_all, base_hash_all, ids_all, ids_all))
    df_list = pool.map(calc_hash, params)

    df = None
    for temp_df, color in zip(df_list, colors):
        if df is None:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on=['Train', 'Extra'], how='inner')
            df = df[df['Train'] != df['Extra']]

    print(df.shape)
    df.to_csv('output/duplicates.csv.gz', index=False, compression='gzip')

    print('\nsuccess!')
