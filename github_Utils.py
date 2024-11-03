import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from tensorflow.keras import backend as K
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.utils import get_custom_objects

import GANs


# %% drawing function.
def draw_learning_curves(history, sub):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy - subject: ' + str(sub))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss - subject: ' + str(sub))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()


def draw_all_learning_curves(history_path):
    # 从文件中加载训练历史
    with open(history_path, 'rb') as f:
        histories = pickle.load(f)

    n_subjects = len(histories)
    plt.figure(figsize=(12, 8))

    for sub in range(n_subjects):
        plt.plot(histories[sub].history['val_accuracy'], label=f'Subject {sub + 1} val_accuracy', linestyle='-',
                 alpha=0.7)
        plt.plot(histories[sub].history['val_loss'], label=f'Subject {sub + 1} val_loss', linestyle='--', alpha=0.7)

    plt.title('Validation Accuracy and Loss for all subjects')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.show()
    plt.close()


def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    # Generate confusion matrix plot
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                  display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + sub)
    plt.savefig(results_path + '/subject_' + sub + '.png')
    plt.show()


def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub + 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model ' + label + ' per subject')
    ax.set_ylim([0, 1])


# %% Set GPU index.
def set_gpu_device(gpu_idx):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs", "Using cuda:", gpu_idx)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU devices available")


# %% Create another initializer.
def cecotti_normal(shape, dtype=None, partition_info=None):
    """
    Initializer proposed by Cecotti et al. 2011:
    https://ieeexplore.ieee.org/document/5492691
    """
    if len(shape) == 1:
        fan_in = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
    else:
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size

    return K.random_normal(shape, mean=0.0, stddev=(1.0 / fan_in))


# %% Create another activation function.
def scaled_tanh(z):
    """
    Scaled hyperbolic tangent activation function, as proposed
    by Lecun 1989: https://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf

    See also Lecun et al. 1998: https://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    return 1.7159 * K.tanh((2.0 / 3.0) * z)


get_custom_objects().update({'scaled_tanh': scaled_tanh})


# %% Create another regularizing.
def streg(a):
    return 0.01 * K.sum(K.square(a[:, 1:, :] - a[:, :-1, :]))


# %% Get GANs models
def get_GANs(X_train, y_train, SubjectIndex, SamplingFrequency, NumAugEpochs, AugmentRate, DataAugment='GANX'):

    # Select the model
    if DataAugment == 'GANX':
        target_data_augment = GANs.GANX(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
    elif DataAugment == 'WGAN':
        target_data_augment, generator_AB = GANs.WGAN(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
    elif DataAugment == 'VAE':
        target_data_augment, generator_AB = GANs.VAE(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
    elif DataAugment == 'TridentGAN':
        target_data_augment = GANs.TridentGAN(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
    elif DataAugment == 'DCGAN':
        target_data_augment, generator_AB = GANs.DCGAN(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
    elif DataAugment == 'CycleGAN':
        target_data_augment, generator_AB = GANs.CycleGAN(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
    elif DataAugment == 'CycleGANX':
        target_data_augment = GANs.CycleGANX(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
    elif DataAugment == 'DiscoGAN':
        target_data_augment, generator_AB = GANs.DiscoGAN(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
        # generator_AB.save(f"DiscoGAN_generator_AB_subject_{SubjectIndex}.h5")
    elif DataAugment == 'DiscoGANX':
        target_data_augment = GANs.DiscoGANX(
            OriginalData=X_train,
            OriginalLabel=y_train,
            SubjectIdx=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate
        )
    else:
        raise Exception("'{}' augment method is not supported yet!".format(DataAugment))

    return target_data_augment, generator_AB
