import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from github_Utils import set_gpu_device, get_GANs
from github_PreProcess import get_data

import shutil
import time
import github_model
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from imblearn.under_sampling import RandomUnderSampler


# %% Get models
def get_model(classification_models, dataset_configuration):
    n_classes = dataset_configuration.get("number_of_classes")
    n_channels = dataset_configuration.get("number_of_channels")
    n_timepoints = dataset_configuration.get("number_of_timepoints")

    # Select the model
    if classification_models == "CNN1":
        model = github_model.CNN1(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif classification_models == "CNN3":
        model = github_model.CNN3(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif classification_models == "CNNR":
        model = github_model.CNNR(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif classification_models == "OCLNN":
        model = github_model.OCLNN(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif classification_models == "BN3":
        model = github_model.BN3(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif classification_models == "EEGInception":
        model = github_model.EEGInception(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif classification_models == "SepConv1D":
        model = github_model.SepConv1D(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif classification_models == "EEGNet":
        model = github_model.EEGNet(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif classification_models == "EEG_DBNet_V2":
        model = github_model.EEG_DBNet_V2(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    else:
        raise Exception("'{}' model is not supported yet!".format(classification_models))

    return model.build_model()


# %% Training
def GANs_train(dataset_configuration, train_configuration, results_path):
    # remove the 'GANsTrainResults' folder before training
    if os.path.exists(results_path):
        # Remove the folder and its contents
        shutil.rmtree(results_path)
        os.makedirs(results_path)

    start_training_time = time.time()
    # Create a file to store the path of the best generator model among all the runs.
    best_generator = open(results_path + "/best_generator.txt", "w")
    # Create a file to store model's performance during training.
    train_log = open(results_path + "/log.txt", "w")

    # Get dataset parameters
    dataset_name = dataset_configuration.get('dataset_name')
    sampling_frequency = dataset_configuration.get('sampling_frequency')

    # Get training hyperparameters
    subject_index = train_configuration.get('subject_index')
    number_of_trains = train_configuration.get('number_of_trains')
    number_of_GAN_training_epochs = train_configuration.get('number_of_GAN_training_epochs')
    number_of_classification_training_epochs = train_configuration.get('number_of_classification_training_epochs')
    classification_training_patience = train_configuration.get('classification_training_patience')
    learning_rate = train_configuration.get('learning_rate')
    batch_size = train_configuration.get('batch_size')
    augment_rate = train_configuration.get('augment_rate')
    GAN_models = train_configuration.get('GAN_models')
    classification_models = train_configuration.get('classification_models')

    # Preparing pretraining models' weight
    with open('../Paper_2/' + classification_models + "/best_models.txt", "r") as best_pretrain_models:
        lines = best_pretrain_models.readlines()
    weight_path = lines[subject_index - 1].strip()
    print("\nUpload the weight of the pretraining models:", weight_path)

    # Preparing original data.
    X_train_origin, y_train_origin, _, _, _, _ = get_data(
        DataSet=dataset_name,
        SubjectIndex=subject_index,
        RandomState=subject_index
    )
    X_train_origin = X_train_origin.squeeze()

    train_validation_data_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=subject_index
    )
    X_train, X_validation, y_train, y_validation = None, None, None, None
    for train_index, validation_index in train_validation_data_split.split(X_train_origin, y_train_origin):
        X_train, X_validation = X_train_origin[train_index], X_train_origin[validation_index]
        y_train, y_validation = y_train_origin[train_index], y_train_origin[validation_index]

    # Let the number of nontarget samples equal target samples.
    X_reshaped = X_validation.reshape((X_validation.shape[0], -1))
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=subject_index)
    X_undersample, y_validation = undersample.fit_resample(X_reshaped, y_validation)
    X_validation = X_undersample.reshape(
        (X_undersample.shape[0], X_validation.shape[1], X_validation.shape[-1])
    )
    y_validation_onehot = to_categorical(y_validation)
    # Don't use the nontarget samples in finetuning that have been used in pretraining.
    pretrain_nontarget_samples = np.load(
        '../Paper_2/' + classification_models + '/pretrain_nontarget_samples.npy'
    )
    nontarget_samples_mask = (y_train == 0)
    not_in_pretrain_mask = np.array(
        [not any(np.array_equal(x, p) for p in pretrain_nontarget_samples) for x in X_train]
    )
    remaining_nontarget_samples = X_train[nontarget_samples_mask & not_in_pretrain_mask]

    # Create folders and files to save a trained generator model.
    file_path = results_path + '/subject_{}'.format(subject_index)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Initialize variables
    acc = np.zeros(number_of_trains)
    kappa = np.zeros(number_of_trains)

    # Initiating variables to save the best subject accuracy among multiple runs.
    BestSubjAcc = 0
    # BestTrainingHistory = []

    print("Training on subject {}...".format(subject_index))
    for train_index in range(number_of_trains):
        start_one_train_time = time.time()
        X_train_target, generator = get_GANs(
            X_train=X_train,
            y_train=y_train,
            SubjectIndex=subject_index,
            SamplingFrequency=sampling_frequency,
            NumAugEpochs=number_of_GAN_training_epochs,
            AugmentRate=augment_rate,
            DataAugment=GAN_models
        )
        generator.save(file_path + '/' + GAN_models + '_generator_run_{}.keras'.format(train_index + 1))
        X_train_nontarget = remaining_nontarget_samples[:len(X_train_target)]
        X_finetune = np.concatenate(([X_train_target, X_train_nontarget]), axis=0)
        y_finetune = np.concatenate(([np.ones(len(X_train_target)), np.zeros(len(X_train_nontarget))]), axis=0)

        X_training = np.concatenate(([X_train, X_finetune]), axis=0)
        y_training = np.concatenate(([y_train, y_finetune]), axis=0)
        y_training_onehot = to_categorical(y_training)

        classification_models_path = file_path + '/classification_run_{}.keras'.format(train_index + 1)

        # Create the model
        model = get_model(classification_models, dataset_configuration)
        model.load_weights('../Paper_2/' + classification_models + weight_path)
        # Compile and train the model
        model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        callbacks = [
            ModelCheckpoint(
                classification_models_path,
                monitor='val_accuracy',
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode='max'
            ),
            # ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001),
            EarlyStopping(
                monitor='val_accuracy',
                verbose=1,
                mode='max',
                patience=classification_training_patience
            )
        ]
        history = model.fit(
            X_training,
            y_training_onehot,
            validation_data=(X_validation, y_validation_onehot),
            epochs=number_of_classification_training_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        model.load_weights(classification_models_path)
        y_pred = model.predict(X_validation).argmax(axis=-1)
        labels = y_validation_onehot.argmax(axis=-1)
        acc[train_index] = accuracy_score(labels, y_pred)
        kappa[train_index] = cohen_kappa_score(labels, y_pred)

        finish_one_train_time = time.time()
        # Print and write performance measures for each run
        training_information = 'Subject: {}   Train no. {}   time: {:.1f} m   '.format(
            subject_index,
            train_index + 1,
            ((finish_one_train_time - start_one_train_time) / 60)
        )
        training_information = training_information + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(
            acc[train_index],
            kappa[train_index]
        )
        print(training_information)
        train_log.write(training_information + '\n')

        # If current training run is better than previous runs, save the history.
        if BestSubjAcc < acc[train_index]:
            BestSubjAcc = acc[train_index]
            BestTrainingHistory = history

    best_run = np.argmax(acc[:])
    best_model_path = file_path + '/' + GAN_models + '_generator_run_{}.keras'.format(best_run + 1) + '\n'
    best_generator.write(best_model_path)

    # Get the current 'OUT' time to calculate the subject training time
    # start_training_time
    finish_training_time = time.time()

    train_information = '\n----------\n'
    train_information = train_information + 'Subject: {}   best_run: {}   Time: {:.1f} h   '.format(
        subject_index,
        best_run + 1,
        (finish_training_time - start_training_time) / (60 * 60)
    )
    train_information = train_information + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(
        acc[best_run],
        np.average(acc[:]),
        acc[:].std()
    )
    train_information = train_information + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}   '.format(
        kappa[best_run],
        np.average(kappa[:]),
        kappa[:].std()
    )
    train_information = train_information + '\n----------'
    print(train_information)
    train_log.write(train_information)

    # Close open files
    best_generator.close()
    train_log.close()


# %% Define run code
def run():
    # Define dataset parameters
    DataSet = 'P300_ALS'

    if DataSet == 'P300_ITBA':
        SamplingFrequency = 250
        NumChannels = 8
        NumSubjects = 8
        NumClasses = 2
        LabelsName = ['Non-target', 'Target']
    elif DataSet == 'P300_ALS':
        SamplingFrequency = 256
        NumChannels = 8
        NumSubjects = 8
        NumClasses = 2
        LabelsName = ['Non-target', 'Target']
    else:
        raise Exception("'{}' dataset is not supported yet!".format(DataSet))

    # Create a folder to store the results of the experiment.
    results_path = os.getcwd() + "/GANsTrainResults"
    if not os.path.exists(results_path):
        # Create a new directory if it does not exist.
        os.makedirs(results_path)

    # Set dataset parameters.
    dataset_configuration = {
        "dataset_name": DataSet,
        "number_of_classes": NumClasses,
        "labels_name": LabelsName,
        "n_subjects": NumSubjects,
        "number_of_channels": NumChannels,
        "number_of_timepoints": SamplingFrequency,
        "sampling_frequency": SamplingFrequency
    }

    # Set training hyperparameters。
    train_configuration = {
        "batch_size": 64,
        "number_of_classification_training_epochs": 10,
        "classification_training_patience": 50,
        "learning_rate": 0.0001,
        "GPU_index": 0,
        "subject_index": 2,
        "number_of_trains": 2,
        "number_of_GAN_training_epochs": 10,
        "augment_rate": 0.1,
        "GAN_models": "DiscoGAN",
        "classification_models": "EEG_DBNet_V2"
    }

    # Choose GPU
    set_gpu_device(train_configuration.get("GPU_index"))

    # Train the model
    GANs_train(dataset_configuration, train_configuration, results_path)


# %%
if __name__ == '__main__':
    run()
