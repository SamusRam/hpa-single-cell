import sys
import h5py

from src.models.encodings_pretrained import BestfittingEncodingsModel

sys.path.insert(0, '..')
import argparse
import pickle
import pandas as pd
from src.models.layers_bestfitting.loss import *
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--num-folds', default=5, type=int)
parser.add_argument('--fold-single', default=None, type=int)


def main():
    args = parser.parse_args()

    num_folds = args.num_folds
    fold_single = args.fold_single

    with open('input/imagelevel_folds_obvious_staining_5.pkl', 'rb') as f:
        folds = pickle.load(f)

    embs_output = 'output/densenet121_embs'

    folds_list = [fold_single] if fold_single is not None else list(range(num_folds))
    all_embs_list = []

    available_basenames = {basename.replace('.h5', '') for basename in os.listdir(embs_output)}
    for fold in folds_list:
        _, val_img_paths = folds[fold]

        fold_img_paths = [os.path.basename(path) for path in val_img_paths
                          if os.path.basename(path) in available_basenames]

        for basename in tqdm(fold_img_paths, desc=f'Processing fold {fold}'):
            try:
                image_level_embs_df = pd.read_hdf(os.path.join(embs_output, f'{basename}.h5'),
                                                  key='data')
            except TypeError:
                print(f'Problem reading {basename}.h5')
            image_level_embs_df.set_index([pd.Index([basename] * len(image_level_embs_df)),
                                             image_level_embs_df['img_cell_number']], inplace=True)
            image_level_embs_df.drop('img_cell_number', axis=1, inplace=True)
            all_embs_list.append(image_level_embs_df)

    all_embs_df = pd.concat(all_embs_list)
    all_embs_df.to_parquet('output/densenet121_embs.parquet')


if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
