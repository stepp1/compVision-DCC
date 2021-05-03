"""
Clothing Dataset Module written in Tensorflow

Author: Victor Faraggi
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def parse_function(filename, label, target_size=224):
    """
    Image parsing function.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized_image = tf.image.resize(image, [target_size, target_size])
    return resized_image, label


def train_preprocess(image, label, seed, target_size=224, normalize_0_1=True):
    """
    Image preprocessing function that applies augmentation.
    """

    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # normalize 0,1 or -1,1
    if normalize_0_1:
        image = image / 255.0
    else:
        image = image / 127.5 - 1

    # # standarize
    # for i in range(3):
    #     # tf images are [w, h, c]
    #     image = image[..., i] - mean[i]
    #     image = image[..., i] / std[i]

    image = tf.image.random_crop(image, [target_size, target_size, 3])

    # image = tf.image.random_central_crop(image, np.random.uniform(0.7, 1.00))

    image = tf.image.stateless_random_flip_left_right(image, seed=seed)

    image = tf.image.stateless_random_brightness(
        image, max_delta=150.0 / 255.0, seed=seed)

    image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed)
    # Make sure the image is still in [0, 1]
    if normalize_0_1:
        image = tf.clip_by_value(image, 0.0, 1.0)
    else:
        image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


class ClothingSmall:
    """
    Clothing Small Dataset
    """
    # pylint: disable=too-many-instance-attributes
    # A dataset has many attrs.
    def __init__(self, data_dir, batch_size=64, img_size=224):
        self.data_dir = Path(data_dir)

        self._prepared = False
        self.batch_size = batch_size
        self.img_size = (img_size, img_size)
        self.input_shape = (img_size, img_size, 3)

        # Paths
        self._train_pth_label = list()
        self._test_pth_label = list()

        # Files
        self._train_files = list()
        self._test_files = list()

        # Labels
        self._train_labels = list()
        self._test_labels = list()

        # Indexes
        self._train_indexes = None
        self._test_indexes = None

        # TF Datasets
        self.mapping = None
        self.num_classes = None

        self.train_ds = None
        self.test_ds = None

    def prepare(self):
        """
        Prepares dataset for use
        """
        self._train_pth_label, self._test_pth_label = self.parse_files(self.data_dir
                                                                       )

        self._train_files, self._test_files = ClothingSmall.process_files(
            train=self._train_pth_label,
            test=self._test_pth_label,
            data_pth=self.data_dir
        )

        self.mapping = self.process_mapping(self.data_dir)
        self.num_classes = len(self.mapping)

        self._train_labels, self._test_labels = self.process_labels(
            self._train_pth_label, self._test_pth_label, categorical=True, num_classes=self.num_classes
        )

        self._train_indexes = self.process_indexes(self._train_labels)
        self._test_indexes = self.process_indexes(self._test_labels)

        self._prepared = True

    def make_ds(self, parse_func, train_preprocess_func, mode='train'):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')

        with tf.device('/cpu:0'):
            if mode == 'train':
                # Create tf dataset, parse, shuffle and apply data augmentation
                train_ds = tf.data.Dataset.from_tensor_slices(
                    ([str(f_name)
                     for f_name in self._train_files], self._train_labels)
                )

                train_ds = train_ds.shuffle(len(self._train_files), reshuffle_each_iteration=True)
                train_ds = train_ds.map(
                    parse_func, num_parallel_calls=AUTOTUNE)
                train_ds = train_ds.map(
                    train_preprocess_func, num_parallel_calls=AUTOTUNE)

                train_ds = train_ds.batch(self.batch_size)
                self.train_ds = train_ds.prefetch(AUTOTUNE)

            if mode == 'train' or mode == 'test':
                # Create tf dataset, parse and don't shuffle nor apply data augmentation
                test_ds = tf.data.Dataset.from_tensor_slices(
                    ([str(f_name)
                     for f_name in self._test_files], self._test_labels)
                )

                test_ds = test_ds.map(parse_func, num_parallel_calls=AUTOTUNE)

                test_ds = test_ds.batch(self.batch_size)
                self.test_ds = test_ds.prefetch(AUTOTUNE)

    def parse_txt(self):
        return ClothingSmall.parse_files(self.data_dir)

    def parse_mapping(self):
        return ClothingSmall.process_mapping(self.data_dir)

    """
    Public Accessors. Most of them only work when dataset has been _prepared.
    """
    def get_files(self, test=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')

        if test:
            return self._train_files, self._test_files

        return self._train_files

    def get_indexes(self, test=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')

        if test:
            return self._train_indexes, self._test_indexes

        return self._train_indexes

    def get_paths(self, test=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')

        if test:
            return self._train_pth_label, self._test_pth_label

        return self._train_pth_label

    def get_labels(self, test=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')
        
        if test:
            return self._train_labels, self._test_labels

        return self._train_labels

    """
    Static Methods used for parsing files
    """

    @staticmethod
    def parse_files(data_pth):
        train_txt_pth = list(data_pth.glob('train*txt'))
        test_txt_pth = list(data_pth.glob('test*txt'))

        if len(train_txt_pth) > 1 or len(test_txt_pth) > 1:
            raise ValueError("There's more than one .txt for Train or Test.")
        else:
            train_txt_pth = train_txt_pth[0]
            test_txt_pth = test_txt_pth[0]

        # use lists to manage indexes easier
        train_pth_label = []
        test_pth_label = []

        with open(train_txt_pth) as train_txt:
            for line in train_txt:
                pth, label = [i.strip() for i in line.strip().split('\t')]

                train_pth_label.append((pth, label))

        with open(test_txt_pth) as test_txt:
            for line in test_txt:
                pth, label = [i.strip() for i in line.strip().split('\t')]

                test_pth_label.append((pth, label))

        return train_pth_label, test_pth_label

    @staticmethod
    def process_mapping(data_pth):
        mapping_pth = list(data_pth.glob('mapping*txt'))

        if len(mapping_pth) > 1:
            raise ValueError("There's more than one mapping.txt")
        else:
            mapping_pth = mapping_pth[0]

        encoding_to_label = dict()

        with open(mapping_pth) as mapping_txt:
            for line in mapping_txt:
                label, code = [i.strip() for i in line.strip().split('\t')]

                encoding_to_label[code] = label

        return encoding_to_label

    """
    Static Methods used for data preparation
    """

    @staticmethod
    def process_indexes(train, test=None):
        # get train/test indexes
        train_indexes = np.arange(len(train))

        if test is not None:
            test_indexes = np.arange(len(test))
            return train_indexes, test_indexes

        return train_indexes

    @staticmethod
    def process_files(train, test=None, data_pth=None):
        base_pth = Path('.')
        if data_pth is not None:
            base_pth = data_pth

        train_files = [base_pth / pth for pth, label in train]

        if test is not None:
            test_files = [base_pth / pth for pth, label in test]
            return train_files, test_files

        return train_files

    @staticmethod
    def process_labels(train, test=None, categorical=False, num_classes=None):
        train_labels = [int(label) for pth, label in train]

        if categorical and num_classes is not None:
            train_labels = tf.keras.utils.to_categorical(
                train_labels, num_classes)

        if test is not None:
            test_labels = [int(label) for pth, label in test]

            if categorical and num_classes is not None:
                test_labels = tf.keras.utils.to_categorical(
                    test_labels, num_classes)

            return train_labels, test_labels

        return train_labels
