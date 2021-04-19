import os
import random

import cv2
import torch
from keras.utils import Sequence
from numpy.random import seed
from torch.utils.data.dataset import Dataset

seed(10)
import numpy as np
from torch.utils.data.sampler import Sampler
from random import sample, shuffle
from .utils import get_cells_from_img, get_cell_img, get_cell_img_with_mask
from multiprocessing import Pool, cpu_count
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from torchvision import transforms
from random import choices
import albumentations as A


class DataGeneneratorRGB(Sequence):
    def __init__(self, paths,
                 batch_size, resized_height, resized_width,
                 basepath_2_y=None,
                 minority_replication_factor=5,
                 balance_classes=True, shuffle=True, augmentation=None,
                 num_channels=3):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.is_test = basepath_2_y is None
        self.paths = paths
        self.colors = ['red', 'green', 'blue']

        self.balance_classes = balance_classes
        self.basepath_2_y = basepath_2_y
        if self.balance_classes and not self.is_test:
            self.minority_replication_factor = minority_replication_factor

            pos_paths = [path for path in paths if basepath_2_y[path] == 1]
            number_of_pos = len(pos_paths)
            self.large_class_portion_size = self.minority_replication_factor * number_of_pos
            self.large_class_i = 0
            self.large_class_vals = [path for path in paths if basepath_2_y[path] == 0]

            self.pos_class_vals = []
            for _ in range(int(minority_replication_factor)):
                self.pos_class_vals += pos_paths
            self.pos_class_vals += sample(pos_paths, int(len(pos_paths) * (minority_replication_factor % 1)))
            self.on_epoch_start()
        else:
            self.current_paths = list(paths)
        self.len = len(self.current_paths) // self.batch_size
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.num_channels = num_channels

        if not shuffle and not self.is_test:
            self.labels = [basepath_2_y[img] for img in self.current_paths[:self.len * self.batch_size]]

    def __len__(self):
        return self.len

    def on_epoch_start(self):
        if self.balance_classes:
            self.current_paths = []

            # negs
            class_idx = self.large_class_i
            class_paths = self.large_class_vals[class_idx: class_idx + self.large_class_portion_size]
            if len(class_paths) < self.large_class_portion_size:
                assert (class_idx + self.large_class_portion_size) % len(
                    self.large_class_vals) > 0, 'Dataloader: index cycle error'
                class_idx = (class_idx + self.large_class_portion_size) % len(self.large_class_vals)
                class_paths += self.large_class_vals[:class_idx]
                self.large_class_i = class_idx
            else:
                self.large_class_i += self.large_class_portion_size

            self.current_paths += class_paths

            self.current_paths += self.pos_class_vals
            self.len = len(self.current_paths) // self.batch_size
        if self.shuffle:
            random.shuffle(self.current_paths)

    # open_rgby adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
    def open_rgby(self, image_path):  # a function that reads RGBY image
        img = [cv2.resize(cv2.imread(f'{image_path}_{color}.png', cv2.IMREAD_GRAYSCALE),
                          (self.resized_height, self.resized_width))
               for color in self.colors]
        img_resized = np.stack(img, axis=-1)
        return img_resized

    def __getitem__(self, idx):
        current_batch = self.current_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels), dtype='float32')

        if not self.is_test:
            y = np.empty(self.batch_size)

        for i, image_path in enumerate(current_batch):
            img = self.open_rgby(image_path)
            if not self.augmentation is None:
                augmented = self.augmentation(image=img)
                img = augmented['image']
            X[i, :, :, :] = img  # preprocess_input(img).astype(np.float32)
            if not self.is_test:
                y[i] = self.basepath_2_y[image_path]
        if not self.is_test:
            return X, y
        return X

    def get_labels(self):
        if self.shuffle:
            image_paths_current = self.current_paths[:self.len * self.batch_size]
            labels = [self.basepath_2_y[img] for img in image_paths_current]
        else:
            labels = self.labels
        return np.array(labels)


class ProteinMLDatasetModified(Dataset):
    def __init__(self,
                 id_list,
                 img_size=1024,
                 in_channels=4,
                 folder='.',
                 resize=True
                 ):
        self.img_size = img_size
        self.in_channels = in_channels
        self.id_list = id_list
        self.folder = folder
        self.resize = resize

        self.num = len(self.id_list)

    def read_rgby(self, image_id):
        if self.in_channels == 3:
            colors = ['red', 'green', 'blue']
        else:
            colors = ['red', 'green', 'blue', 'yellow']

        flags = cv2.IMREAD_GRAYSCALE
        if self.resize:
            img = [cv2.resize(cv2.imread(os.path.join(self.folder, f'{image_id}_{color}.png'), flags),
                              (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                   for color in colors]
        else:
            img = [cv2.imread(os.path.join(self.folder, f'{image_id}_{color}.png'), flags)
                   for color in colors]
        img = np.stack(img, axis=-1)
        img = img / 255.0
        return img

    def __getitem__(self, index):
        image_id = self.id_list[index]

        image = self.read_rgby(image_id)

        return image

    def __len__(self):
        return self.num


class ProteinDatasetImageLevel(Dataset):
    def __init__(self,
                 img_paths,
                 basepath_2_ohe=None,
                 img_size=1024,
                 transform=None,
                 return_label=True,
                 is_trainset=True,
                 in_channels=4,
                 crop_size=0,
                 random_crop=False,
                 cherrypicked_mitotic_spindle_df=None,
                 cherrypicked_aggresome_df=None,
                 mitotic_img_prob=None,
                 max_num_mitotic_cells_per_img=None,
                 aggresome_img_prob=None,
                 max_num_aggresome_cells_per_img=None
                 ):

        self.is_trainset = is_trainset
        self.img_size = img_size
        self.return_label = return_label
        self.in_channels = in_channels
        self.transform = transform
        self.crop_size = crop_size
        self.random_crop = random_crop

        self.img_paths = img_paths
        if is_trainset:
            self.basepath_2_ohe = basepath_2_ohe

        self.mitotic_aggresome_balancing = cherrypicked_aggresome_df is not None
        if self.mitotic_aggresome_balancing:
            assert cherrypicked_mitotic_spindle_df is not None, 'when balancing minor classes, both Aggresome and Mitotic must be included'
            self.cherrypicked_aggresome_df = cherrypicked_aggresome_df
            self.cherrypicked_mitotic_spindle_df = cherrypicked_mitotic_spindle_df

            self.minority_aug = A.Compose([
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.7),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(p=0.8, rotate_limit=0),
                A.GaussianBlur(p=0.2)])
            self.indices_of_mitotic_cells = list(range(len(cherrypicked_mitotic_spindle_df)))
            self.indices_of_aggresome_cells = list(range(len(cherrypicked_aggresome_df)))
            self.mitotic_img_prob = 0.1 if mitotic_img_prob is None else mitotic_img_prob
            self.aggresome_img_prob = 0.1 if aggresome_img_prob is None else aggresome_img_prob
            self.max_num_mitotic_cells_per_img = 4 if max_num_mitotic_cells_per_img is None else max_num_mitotic_cells_per_img
            self.max_num_aggresome_cells_per_img = 4 if max_num_aggresome_cells_per_img is None else max_num_aggresome_cells_per_img

            self.img_paths_len = len(self.img_paths)
            self.num = self.img_paths_len
        else:
            self.num = len(self.img_paths)

    def copy_paste_augment(self, img_rgby, is_aggresome):
        selected_cells_df = self.cherrypicked_aggresome_df if is_aggresome else self.cherrypicked_mitotic_spindle_df
        indices_of_selected_cells = self.indices_of_aggresome_cells if is_aggresome else self.indices_of_mitotic_cells
        max_num_cells_per_img = self.max_num_aggresome_cells_per_img if is_aggresome else self.max_num_mitotic_cells_per_img

        number_of_added_cells = np.random.randint(max(1, max_num_cells_per_img // 2), max_num_cells_per_img)
        img_rgby_height, img_rgby_width = img_rgby.shape[:2]

        sampling_weights = selected_cells_df['sampling_weight']
        ohes = []
        for mitotic_cell_idx in choices(indices_of_selected_cells, weights=sampling_weights, k=number_of_added_cells):
            cell_img, cell_mask = get_cell_img_with_mask(selected_cells_df['ID'].iloc[mitotic_cell_idx],
                                                         selected_cells_df['cell_i'].iloc[mitotic_cell_idx],
                                                         selected_cells_df['is_public'].iloc[mitotic_cell_idx])

            augmented = self.minority_aug(image=cell_img, mask=cell_mask)

            cell_img = augmented['image']
            cell_mask = augmented['mask']

            cell_rows, cell_cols = np.where(cell_mask)
            cell_img_height, cell_img_width = cell_mask.shape
            x_insert = np.random.randint(0, img_rgby_width - cell_img_width)
            y_insert = np.random.randint(0, img_rgby_height - cell_img_height)

            img_cell_rows = cell_rows + y_insert
            img_cell_cols = cell_cols + x_insert

            img_rgby[img_cell_rows, img_cell_cols] = cell_img[cell_rows, cell_cols]

            ohes.append(selected_cells_df['ohe'].iloc[mitotic_cell_idx])
        return img_rgby, ohes

    def get_tiled_cell(self, is_aggresome):
        selected_cells_df = self.cherrypicked_aggresome_df if is_aggresome else self.cherrypicked_mitotic_spindle_df
        indices_of_selected_cells = self.indices_of_aggresome_cells if is_aggresome else self.indices_of_mitotic_cells

        img_rgby_height, img_rgby_width = self.img_size, self.img_size

        sampling_weights = selected_cells_df['sampling_weight']
        mitotic_cell_idx = choices(indices_of_selected_cells, weights=sampling_weights, k=1)[0]
        cell_img = get_cell_img_with_mask(selected_cells_df['ID'].iloc[mitotic_cell_idx],
                                          selected_cells_df['cell_i'].iloc[mitotic_cell_idx],
                                          selected_cells_df['is_public'].iloc[mitotic_cell_idx],
                                          return_mask=False)


        cell_img = self.minority_aug(image=cell_img)['image']

        height, width = cell_img.shape[:2]

        cell_img_tiled = np.tile(cell_img, [img_rgby_height // height + 1, img_rgby_width // width + 1, 1])
        cell_img_tiled = cell_img_tiled[:img_rgby_height, :img_rgby_height, :]

        return cell_img_tiled, selected_cells_df['ohe'].iloc[mitotic_cell_idx]

    def read_rgby(self, index):
        if self.in_channels == 3:
            colors = ['red', 'green', 'blue']
        else:
            colors = ['red', 'green', 'blue', 'yellow']
        img_path = self.img_paths[index]

        img_id = os.path.basename(img_path)
        img = [cv2.resize(cv2.imread(f'{img_path}_{color}.png', cv2.IMREAD_GRAYSCALE), (self.img_size, self.img_size))
               for color in colors]
        img_rgby = np.stack(img, axis=-1)
        label = self.basepath_2_ohe[img_path]
        return img_rgby, label, img_id

    def get_rgby(self, index):
        if not self.mitotic_aggresome_balancing:
            return self.read_rgby(index)

        # if index < self.img_paths_len:
        #     mitotic_aggresome_is_balanced = False
        #     tiled_minority = False
        # elif index < 2*self.img_paths_len:
        #     mitotic_aggresome_is_balanced = True
        #     tiled_minority = False
        #     index = index % self.img_paths_len
        # else:
        #     raise NotImplementedError('Not supported mitotic-aggresome aug')
        #     mitotic_aggresome_is_balanced = False
        #     tiled_minority = True

        # if not tiled_minority:
        #     img_rgby, label, img_id = self.read_rgby(index)
        #
        #     additional_ohes = []
        #     if mitotic_aggresome_is_balanced:
        #         if np.random.rand() < self.mitotic_img_prob:
        #             img_rgby, ohes_of_added = self.copy_paste_augment(img_rgby, is_aggresome=False)
        #             additional_ohes.extend(ohes_of_added)
        #         if np.random.rand() < self.aggresome_img_prob:
        #             img_rgby, ohes_of_added = self.copy_paste_augment(img_rgby, is_aggresome=True)
        #             additional_ohes.extend(ohes_of_added)
        #     if len(additional_ohes):
        #         label = label.copy()
        #         for additional_ohe in additional_ohes:
        #             label += additional_ohe
        #         label = np.minimum(1, label)
        # else:
        #     img_id = 'tiled'
        #     img_rgby, label = self.get_tiled_cell(is_aggresome=np.random.rand() < 0.5)

        if np.random.rand() < self.mitotic_img_prob:
            img_id = 'tiled'
            img_rgby, label = self.get_tiled_cell(is_aggresome=False)
        elif np.random.rand() < self.aggresome_img_prob:
            img_id = 'tiled'
            img_rgby, label = self.get_tiled_cell(is_aggresome=True)
        else:
            img_rgby, label, img_id = self.read_rgby(index)
        return img_rgby, label, img_id

    def __getitem__(self, index):
        image, label, img_id = self.get_rgby(index)

        if self.transform is not None:
            image = self.transform(image)
        image = image / 255.0
        image = image.astype(np.float32)
        if len(image.shape) == 3:
            image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        if self.return_label:
            return image, label, img_id
        else:
            return image, img_id

    def __len__(self):
        return self.num


class ProteinDatasetCellLevel(Dataset):
    def __init__(self,
                 img_paths,
                 labels_df=None,
                 img_size=512,
                 batch_size=32,
                 transform=None,
                 return_label=True,
                 is_trainset=True,
                 in_channels=4,
                 ):
        self.is_trainset = is_trainset
        self.img_size = img_size
        self.return_label = return_label
        self.in_channels = in_channels
        self.transform = transform
        self.batch_size = batch_size

        labeled_paths_set = set(labels_df.index.get_level_values(0))
        self.img_paths = [img_path for img_path in img_paths if img_path in labeled_paths_set]

        self.num = len(self.img_paths)
        if is_trainset:
            self.labels_df = labels_df

    def preprocess_image(self, image):
        if self.transform is not None:
            image = self.transform(image)
        image = image / 255.0
        image = image.astype(np.float32)
        if len(image.shape) == 3:
            image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image

    def __getitem__(self, index):
        img_basepath = self.img_paths[index]
        if self.is_trainset:
            cell_labels_df = self.labels_df.loc[img_basepath]
            cell_imgs, cell_labels = get_cells_from_img(img_basepath,
                                                        cell_img_size=self.img_size, return_raw=False,
                                                        sample_size=self.batch_size, cell_labels_df=cell_labels_df)
            cell_labels_np = np.stack(cell_labels)
        else:
            cell_imgs = get_cells_from_img(img_basepath,
                                           cell_img_size=self.img_size, return_raw=False,
                                           sample_size=self.batch_size)

        cell_imgs = [self.preprocess_image(img) for img in cell_imgs]
        cell_imgs_np = np.stack(cell_imgs)

        if self.is_trainset:
            return cell_imgs_np, cell_labels_np, index
        return cell_imgs_np, index

    def __len__(self):
        return self.num


def img_nmi_avg_parellel(preds_all):
    img_path = preds_all.index[0][0]
    preds_all = preds_all.values
    nmis_pairwise = []
    for i in range(len(preds_all)):
        for j in range(i + 1, len(preds_all)):
            nmi_ = (normalized_mutual_info_score(preds_all[j][0],
                                                 preds_all[i][0]) + normalized_mutual_info_score(
                preds_all[i][0], preds_all[j][0])) / 2
            nmis_pairwise.append(nmi_)
    return pd.DataFrame({'img_path': [img_path], 'similarity': [np.nanmean(nmis_pairwise)]})


def applyParallel(dfGrouped, func):
    with Pool(cpu_count() // 2) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)


class ProteinDatasetCellSeparateLoading(Dataset):
    def __init__(self,
                 img_paths,
                 labels_df=None,
                 img_size=512,
                 transform=None,
                 return_label=True,
                 in_channels=4,
                 hard_data_subsample=None,
                 int_labels=False,
                 image_level_labels=False,
                 basepath_2_ohe=None,
                 normalize=False
                 ):
        self.img_size = img_size
        self.return_label = return_label
        self.in_channels = in_channels
        self.transform = transform

        labeled_paths_set = set(labels_df.index.get_level_values(0))
        img_paths = [img_path for img_path in img_paths if img_path in labeled_paths_set]

        labels_df = labels_df.loc[img_paths]

        if hard_data_subsample is not None:
            df_proxy_variability = applyParallel(labels_df.groupby(level=0), img_nmi_avg_parellel)

            num_samples = int(np.ceil(hard_data_subsample * len(df_proxy_variability)))

            hard_paths = set(df_proxy_variability.sort_values(by='similarity')['img_path'].values[:num_samples])
            labeled_paths_all = [path for path in labels_df.index.get_level_values(0) if path not in hard_paths]
            # adding random sample out of other imgs not to focus on potentially weird imgs with low similarity only
            labeled_paths_all = sample(labeled_paths_all, max(1, num_samples)) + list(hard_paths)
            labels_df = labels_df.loc[set(labeled_paths_all)]

        self.num = len(labels_df)
        self.labels_df = labels_df
        self.int_labels = int_labels
        self.basepath_cell = labels_df.index.values
        self.image_level_labels = image_level_labels
        self.basepath_2_ohe_vector = basepath_2_ohe
        self.normalize = normalize
        if self.normalize:
            self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.406],
                                                      std=[0.229, 0.224, 0.225, 0.225])

    def preprocess_image(self, image):
        image = image / 255.0
        if len(image.shape) == 3:
            image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        if self.normalize:
            image = self.normalization(image)
        return image

    def __getitem__(self, index):
        img_basepath, cell_i = self.basepath_cell[index]
        if self.image_level_labels:
            y = self.basepath_2_ohe_vector[img_basepath]
        else:
            y_raw = self.labels_df.loc[(img_basepath, cell_i), 'image_level_pred']
            if self.int_labels:
                random_numbers = np.random.uniform(size=len(y_raw))
                y = np.zeros_like(y_raw)
                y[random_numbers < y_raw] = 1
            else:
                y = y_raw

        cell_img = get_cell_img(img_basepath, cell_i, aug=self.transform)

        cell_img = self.preprocess_image(cell_img)

        return cell_img, y, index

    def __len__(self):
        return self.num


class BalancingSubSampler(Sampler[int]):

    def __init__(self, trn_img_paths, basepath_2_ohe_vector, class_names, required_class_count=1500) -> None:
        self.trn_ohes = np.array([basepath_2_ohe_vector[path] for path in trn_img_paths])
        self.class_name_2_i = {name: i for i, name in enumerate(class_names)}
        self.class_name_2_count = [(name, count) for name, count in zip(class_names,
                                                                        self.trn_ohes.sum(axis=0))]
        self.class_name_2_indices = {class_name: np.where(self.trn_ohes[:, class_i] == 1)[0]
                                     for class_i, class_name in enumerate(class_names)}
        self.required_class_count = required_class_count
        self.selected_indices = None

    def prepare_balanced_subset(self):
        shuffle(self.class_name_2_count)
        self.selected_indices = []
        selected_indices_set = set()
        for class_name, class_count in self.class_name_2_count:
            # check already added
            present_count = self.trn_ohes[self.selected_indices, self.class_name_2_i[class_name]].sum()
            needed_additionally_count = max(0, self.required_class_count - present_count)

            remaining_class_indices = [idx for idx in self.class_name_2_indices[class_name]
                                       if idx not in selected_indices_set]
            if len(remaining_class_indices) > needed_additionally_count:
                class_indices_added = sample(remaining_class_indices, needed_additionally_count)
            else:
                class_indices_added = remaining_class_indices

            self.selected_indices.extend(class_indices_added)
            selected_indices_set.update(class_indices_added)

    @property
    def num_samples(self) -> int:
        return len(self.selected_indices)

    def __iter__(self):
        iterator = iter(self.selected_indices)
        return iterator

    def __len__(self):
        self.prepare_balanced_subset()
        return self.num_samples
