from pathlib import Path

import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

def parse_function(filename, label, target_size=224):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    resized_image = tf.image.resize(image, [target_size, target_size])
    return resized_image, label

def train_preprocess(image, label, seed, target_size=224):
    image = tf.image.central_crop(image, central_fraction = 0.7)
    
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    
    # Back to the original size
    image = tf.image.resize(image, size=[target_size, target_size])
    
    image = tf.image.stateless_random_flip_left_right(image, seed=new_seed)

    image = tf.image.stateless_random_brightness(
        image, max_delta=32.0 / 255.0, seed=new_seed)
    image = tf.image.stateless_random_saturation(
        image, lower=0.5, upper=1.5, seed=new_seed)

    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


class ClothingSmall(object):
    def __init__(self, data_dir, bs=64, img_size=224):
        self.DATA_DIR = Path(data_dir)

        self._prepared = False
        self.bs = bs
        self.img_size = (img_size, img_size)
        self.input_shape = (img_size, img_size, 3)
        
        # Paths
        self._train_pth_label = None
        self._test_pth_label = None
        
        # Files
        self._train_files = None
        self._test_files = None
        
        # Labels
        self._train_labels = None
        self._test_labels = None

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
        self._train_pth_label, self._test_pth_label = self.parse_txt(self.DATA_DIR
        )

        self._train_files, self._test_files = ClothingSmall.get_files(
            self._train_pth_label, 
            self._test_pth_label, 
            self.DATA_DIR
            )

        self.mapping = self.parse_mapping(self.DATA_DIR)
        self.num_classes = len(self.mapping)

        self._train_labels, self._test_labels = ClothingSmall.get_labels(
            self._train_pth_label, self._test_pth_label, categorical=True, num_classes=self.num_classes
            )

        

        self._train_indexes = ClothingSmall.get_indexes(self._train_labels)
        self._test_indexes = ClothingSmall.get_indexes(self._test_labels)

        self._prepared = True

    def make_ds(self, parse_func, train_preprocess_func, test=True, test_only=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')
            
        if not test_only:
            # Create tf dataset, parse, shuffle and apply data augmentation
            train_ds = tf.data.Dataset.from_tensor_slices(
                ([str(f_name) for f_name in self._train_files], self._train_labels)
            )

            train_ds = train_ds.shuffle(len(self._train_files))
            train_ds = train_ds.map(parse_func, num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.map(train_preprocess_func, num_parallel_calls=AUTOTUNE)

            train_ds = train_ds.batch(self.bs)
            self.train_ds = train_ds.prefetch(1)

        if test or test_only:
            # Create tf dataset, parse and don't shuffle nor apply data augmentation
            test_ds = tf.data.Dataset.from_tensor_slices(
                ([str(f_name) for f_name in self._test_files], self._test_labels)
            )

            test_ds = test_ds.map(parse_func, num_parallel_calls=AUTOTUNE)

            test_ds = test_ds.batch(self.bs)
            self.test_ds = test_ds.prefetch(1)

    def parse_txt(self):
        return ClothingSmall.parse_txt(self.DATA_DIR)

    def parse_mapping(self):
        return ClothingSmall.parse_mapping(self.DATA_DIR)

    """
    Public Accessors. Most of them only work when dataset has been _prepared.
    """

    def get_files(self, train, test=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')
        
        if self.test:
            return self._train_files, self._test_files

        return self.train_files

    def get_indexes(self, train, test=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')
        
        if self.test:
            return self._train_indexes, self._test_indexes

        return self._train_indexes

    def get_paths(self, train, test=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')
        
        if self.test:
            return self._train_pth_label, self._test_pth_label

        return self._train_pth_label

    def get_labels(self, train, test=False):
        if not self._prepared:
            raise RuntimeError('Dataset is not _prepared yet.')
        
        if self.test:
            return self._train_labels, self._test_labels

        return self._train_labels

    """
    Static Methods used for parsing files
    """

    @staticmethod
    def parse_txt(data_pth):
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
    def parse_mapping(data_pth):
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
    def get_indexes(train, test=None):
        # get train/test indexes
        train_indexes = np.arange(len(train))
        
        if test is not None:
            test_indexes = np.arange(len(test))
            return train_indexes, test_indexes
        
        return train_indexes

    @staticmethod
    def get_files(train, test=None, data_pth=None):
        base_pth = Path('.')
        if data_pth is not None:
            base_pth = data_pth
        
        train_files = [base_pth / pth for pth, label in train]
        
        if test is not None:
            test_files = [base_pth / pth for pth, label in test]
            return train_files, test_files
        
        return train_files

    @staticmethod
    def get_labels(train, test=None, categorical=False, num_classes=None):
        train_labels = [int(label) for pth, label in train]

        if categorical and num_classes is not None:
            train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)

        if test is not None:
            test_labels = [int(label) for pth, label in test]

            if categorical and num_classes is not None:
                test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

            return train_labels, test_labels
        
        return train_labels