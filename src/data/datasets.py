import cv2
from keras.utils import Sequence
from torch.utils.data.dataset import Dataset
from numpy.random import seed
import random
from random import sample
import os
seed(10)
import numpy as np


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

        img = [cv2.imread(f'{image_path}_{color}.png', cv2.IMREAD_GRAYSCALE)
               for color in self.colors]
        img_resized = cv2.resize(np.stack(img, axis=-1), (self.resized_height, self.resized_width))
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