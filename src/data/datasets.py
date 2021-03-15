import cv2
from keras.utils import Sequence
import torch
from torch.utils.data.dataset import Dataset
from numpy.random import seed
import random
from random import sample
import os
seed(10)
import numpy as np
from torch.utils.data.sampler import Sampler
from random import sample, shuffle


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
                 folder='.'
                 ):
        self.img_size = img_size
        self.in_channels = in_channels
        self.id_list = id_list
        self.folder = folder

        self.num = len(self.id_list)

    def read_rgby(self, image_id):
        if self.in_channels == 3:
            colors = ['red', 'green', 'blue']
        else:
            colors = ['red', 'green', 'blue', 'yellow']

        flags = cv2.IMREAD_GRAYSCALE
        img = [cv2.resize(cv2.imread(os.path.join(self.folder, f'{image_id}_{color}.png'), flags), (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
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

        self.num = len(self.img_paths)

    def read_rgby(self, img_path):
        if self.in_channels == 3:
            colors = ['red', 'green', 'blue']
        else:
            colors = ['red', 'green', 'blue', 'yellow']

        flags = cv2.IMREAD_GRAYSCALE
        img = [cv2.resize(cv2.imread(f'{img_path}_{color}.png', flags), (self.img_size, self.img_size))
               for color in colors]
        img = np.stack(img, axis=-1)
        return img

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = self.read_rgby(img_path)

        if self.transform is not None:
            image = self.transform(image)
        image = image / 255.0
        image = image.astype(np.float32)
        if len(image.shape) == 3:
            image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        if self.return_label:
            label = self.basepath_2_ohe[img_path]
            return image, label, index
        else:
            return image, index

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