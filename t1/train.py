"""
jsaavedr, 2020
This allows you to train and test your model

Before using this program, set the path where the folder "covnet2"  is stored.
To use train.py, you will require to send the following parameters :
 * -config : A configuration file where a set of parameters for data construction and trainig is set.
 * -name: A section name in the configuration file.
 * -mode: [train, test] for training, testing, or showing  variables of the current model. By default this is set to 'train'
 * -save: Set true for saving the model


 Extension made by Victor Faraggi, 2021

 Added modularity. Now you can import the following functions:
    - create_config(name, config_file=None, config_str=None)
        -> return a ConfigurationFile from config_file path or config_str
    - parse_config(config)
        -> returns dict w/ tfr_files
    - load_dataset(config, tfr_train_file, tfr_test_file)
        -> returns a dict w/ train/test datasets, mean_image, input_shape and number_of_classes
    - create_model(config, model_name, in_shape)
        -> returns a tf model
    - create_scheduler(config)
        -> returns a scheduler
    - create_opt(opt_name, config, scheduler=None)
        -> returns an optimizer
    - create_cbs(config)
        -> returns a TensorBoardCallback and a CheckpointCallback
    - run_model(mode, model, opt, datasets, config, train_cbs=None, test_cbs=None):
        -> returns the training/test history
    - save_model(model, config, fname)
        -> saves the model

    config argument is expected to be a ConfigurationFile
"""
import os
import sys
import pickle

# set the convnet2 path
sys.path.append("/home/step/Personal/UCH/2021-sem1/VisionComp/convnet2")

from models import resnet, alexnet
import datasets.data as data
import utils.configuration as conf
import utils.losses as losses
import numpy as np
import argparse

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}


def create_config(name, config_file=None, config_str=None):
    config_obj = None
    if config_file:
        config_obj = conf.ConfigurationFile(config_file, name)
    elif config_str:
        raise NotImplementedError
    return config_obj


def parse_config(config, mode):
    tfr_file = dict()
    if mode == 'train':
        tfr_train_file = os.path.join(config.get_data_dir(), "train.tfrecords")
        tfr_file['train'] = tfr_train_file
    if mode == 'train' or mode == 'test':
        tfr_test_file = os.path.join(config.get_data_dir(), "test.tfrecords")
        tfr_file['test'] = tfr_test_file
    if config.use_multithreads():
        if mode == 'train':
            tfr_train_file = [os.path.join(config.get_data_dir(), "train_{}.tfrecords".format(idx)) for idx in
                              range(config.get_num_threads())]
            tfr_file['train'] = tfr_train_file
        if mode == 'train' or pargs.mode == 'test':
            tfr_test_file = [os.path.join(config.get_data_dir(), "test_{}.tfrecords".format(idx)) for idx in
                             range(config.get_num_threads())]
            tfr_file['test'] = tfr_test_file

    sys.stdout.flush()
    return tfr_file


def load_dataset(config, tfr_train_file, tfr_test_file, mode):
    mean_file = os.path.join(config.get_data_dir(), "mean.dat")
    shape_file = os.path.join(config.get_data_dir(), "shape.dat")

    dataset = dict()

    input_shape = np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)

    number_of_classes = config.get_number_of_classes()

    dataset['input_shape'] = input_shape
    dataset['mean_image'] = mean_image
    dataset['number_of_classes'] = number_of_classes

    # loading tfrecords into dataset object
    if mode == 'train':
        tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
        tr_dataset = tr_dataset.map(
            lambda x: data.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation=True))
        tr_dataset = tr_dataset.shuffle(config.get_shuffle_size())
        tr_dataset = tr_dataset.batch(batch_size=config.get_batch_size())
        # tr_dataset = tr_dataset.repeat()
        dataset['train'] = tr_dataset

    if mode == 'train' or mode == 'test':
        val_dataset = tf.data.TFRecordDataset(tfr_test_file)
        val_dataset = val_dataset.map(
            lambda x: data.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation=False))
        val_dataset = val_dataset.batch(batch_size=config.get_batch_size())
        dataset['test'] = val_dataset

    return dataset


def create_model(config, model_name, in_shape, only_test_input=None, use_mixed=False):
    if use_mixed:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

    # save_freq = configuration.get_snapshot_steps())
    if only_test_input is None:
        if model_name == 'resnet-34':
            model = resnet.ResNet([3, 4, 6, 3], [64, 128, 256, 512], config.get_number_of_classes(), se_factor=0)
            print('Model is Resnet-34')
        elif model_name == 'resnet-50':
            model = resnet.ResNet([3, 4, 6, 3], [64, 128, 256, 512], config.get_number_of_classes(),
                                  use_bottleneck=True)
            print('Model is Resnet-50')
        elif model_name == 'resnet-50':
            model = alexnet.AlexNetModel(config.get_number_of_classes())
            print('Model is AlexNet')
        else:
            model = resnet.ResNet([3, 4, 6, 3], [64, 128, 256, 512], config.get_number_of_classes(), se_factor=0)
            print('Model is Resnet-34')

        sys.stdout.flush()
    else:
        model = only_test_input

    # build the model indicating the input shape
    # define the model input
    print(in_shape)
    input_image = tf.keras.Input((in_shape[0], in_shape[1], in_shape[2]), name='input_image')
    model(input_image)
    model.summary()
    return model


def create_scheduler(config):
    initial_learning_rate = config.get_learning_rate()
    cosine_scheduler = tf.keras.experimental.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=config.get_decay_steps(),
        alpha=0.0001
    )
    return cosine_scheduler


def create_opt(opt_name, config, scheduler=None):
    if opt_name.upper() == 'SDG':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config.get_learning_rate() if scheduler is None else scheduler,
            momentum=0.9,
            nesterov=True
        )

    elif opt_name.upper() == 'ADAM':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get_learning_rate()
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config.get_learning_rate(),
            momentum=0.9,
            nesterov=True
        )
    return optimizer


def run_model(mode, model, opt, datasets, config, train_cbs=None, test_cbs=None):
    if test_cbs is None:
        test_cbs = []
    if train_cbs is None:
        train_cbs = []

    model.compile(
        optimizer=opt,
        loss=losses.crossentropy_loss,
        metrics=['accuracy']
    )

    if mode == 'train':
        history = model.fit(
            datasets['train'],
            epochs=config.get_number_of_epochs(),
            validation_data=datasets['test'],
            callbacks=train_cbs
        )

        # save history
        saved_to = os.path.join(config.get_data_dir(), 'history.pkl')
        with open(saved_to, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        return history

    elif mode == 'test':
        test_loss = model.evaluate(
            datasets['test'],
            callbacks=test_cbs
        )

        return test_loss


def create_cbs(config):
    # callback
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=config.get_snapshot_dir(),
                                           histogram_freq=1)
    # Defining callback for saving checkpoints
    # save_freq: frecuency in terms of number steps each time checkpoint is saved

    pth = config.get_snapshot_dir() + '{epoch:03d}.h5'
    chk_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=pth,
        save_weights_only=True,
        mode='max',
        monitor='val_acc',
        save_freq='epoch',
    )

    return tb_cb, chk_cb


def save_model(model, config, fname="cnn-model"):
    saved_to = os.path.join(config.get_data_dir(), fname)
    model.save(saved_to)
    print("model saved to {}".format(saved_to))


def main(args):
    configuration_file = args.config
    configuration = create_config(name=args.name, config_file=configuration_file)

    tfr_files = parse_config(configuration, args.mode)
    datasets = load_dataset(configuration, tfr_files['train'], tfr_files['test'])

    # this code allows program to run in  multiple GPUs. It was tested with 2 gpus.
    mirrored = True
    if mirrored:
        tf.debugging.set_log_device_placement(True)
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise NotImplementedError

    with strategy.scope():
        tensorboard_callback, model_checkpoint_callback = create_cbs(configuration)

        if pargs.model == "":
            model = 'resnet-34'
        else:
            model = pargs.model

        model = create_model(configuration, model, datasets['input_shape'])

        # use_checkpoints to load weights
        if configuration.use_checkpoint():
            model.load_weights(configuration.get_checkpoint_file(), by_name=True, skip_mismatch=True)
            # model.load_weights(configuration.get_checkpoint_file(), by_name = False)

        # define optimizer, my experince shows that SGD + cosine decay is a good starting point
        # recommended learning_rate is 0.1, and decay_steps = total_number_of_steps
        lr_scheduler = create_scheduler(configuration)

        opt = create_opt('adam', lr_scheduler)

        _ = run_model(
            mode=pargs.mode,
            model=model,
            opt=opt,
            datasets=datasets,
            config=configuration,
            train_cbs=[model_checkpoint_callback],
            test_cbs=[tensorboard_callback]
        )

        # save the model
        if pargs.save:
            save_model(model, configuration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple mnist model")

    parser.add_argument("-config", type=str, help="<str> configuration file", required=True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required=True)
    parser.add_argument("-mode", type=str, choices=['train', 'test'], help=" train or test", required=False,
                        default='train')
    parser.add_argument("-save", type=bool, help=" True to save the model", required=False, default=False)
    parser.add_argument("-model", type=str, help="Model Name", required=False, default="")

    pargs = parser.parse_args()

    main(pargs)
