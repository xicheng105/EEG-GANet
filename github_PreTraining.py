import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time
import models
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit

from github_PreProcess import get_data
from github_Utils import draw_learning_curves, draw_performance_barChart, draw_confusion_matrix, set_gpu_device


# %% Get models
def get_model(model_name, dataset_conf):
    n_classes = dataset_conf.get("n_classes")
    n_channels = dataset_conf.get("n_channels")
    n_timepoints = dataset_conf.get("n_timepoints")

    # Select the model
    if model_name == "CNN1":
        model = models.CNN1(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif model_name == "CNN3":
        model = models.CNN3(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif model_name == "CNNR":
        model = models.CNNR(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif model_name == "OCLNN":
        model = models.OCLNN(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif model_name == "BN3":
        model = models.BN3(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif model_name == "EEGInception":
        model = models.EEGInception(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif model_name == "SepConv1D":
        model = models.SepConv1D(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif model_name == "EEGNet":
        model = models.EEGNet(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    elif model_name == "EEG_DBNet_V2":
        model = models.EEG_DBNet_V2(NumChannels=n_channels, NumClasses=n_classes, SamplingFrequency=n_timepoints)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model.build_model()


# %% Training
def train(dataset_conf, train_conf, results_path):
    # remove the 'PretrainResults' folder before training
    if os.path.exists(results_path):
        # Remove the folder and its contents
        shutil.rmtree(results_path)
        os.makedirs(results_path)

    # Get the current 'IN' time to calculate the overall training time
    StartTrain = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best_models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")

    # Get dataset parameters
    DataSet = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_subjects')
    Standardization = dataset_conf.get('standardize')
    LOSO = dataset_conf.get('LOSO')
    DataFiltering = dataset_conf.get('data_filtering')
    DropNontargetSample = dataset_conf.get('drop_non-target_sample')
    DataAugmentation = dataset_conf.get('data_augmentation')
    NumAugEpochs = dataset_conf.get('augment_epoch')
    AugmentRate = dataset_conf.get('augment_rate')
    RemainRate = dataset_conf.get('remain_rate')

    # Get training hypermarkets
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('learning_rate')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_trains')
    model_name = train_conf.get('model_name')

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    # Iteration over subjects
    # for sub in range(n_sub-1, n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    for sub in range(n_sub):  # (num_sub): for all subjects, (i-1,i): for the ith subject.

        StartSub = time.time()
        print('\nTraining on subject ', sub + 1)
        log_write.write('\nTraining on subject ' + str(sub + 1) + '\n')

        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0
        BestTrainingHistory = []

        # Get training and test data
        X_train, _, y_train_onehot, _, _, _ = get_data(
            DataSet=DataSet,
            SubjectIndex=sub + 1,
            LOSO=LOSO,
            Standardization=Standardization,
            DataFiltering=DataFiltering,
            DropNontargetSample=DropNontargetSample,
            DataAugment=DataAugmentation,
            RandomState=sub + 1,
            AugmentRate=AugmentRate,
            RemainRate=RemainRate,
            NumAugEpochs=NumAugEpochs
        )

        training_validation_data_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.25
        )
        X_training, y_training_onehot, X_validation, y_validation_onehot = None, None, None, None
        for train_index, valid_index in training_validation_data_split.split(X_train, y_train_onehot):
            X_training, X_validation = X_train[train_index], X_train[valid_index]
            y_training_onehot, y_validation_onehot = y_train_onehot[train_index], y_train_onehot[valid_index]

        # Iteration over multiple runs
        for train_idx in range(n_train):  # How many repetitions of training for subject i.

            # Get the current 'IN' time to calculate the 'run' training time
            StartTrainIdx = time.time()

            # Create folders and files to save trained models for all runs
            filepath = results_path + '/saved models/subject_{}'.format(sub + 1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = filepath + '/run_{}.keras'.format(train_idx + 1)

            # Create the model
            model = get_model(model_name, dataset_conf)
            # Compile and train the model
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0,
                                save_best_only=True, save_weights_only=True, mode='max'),
                # ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_training, y_training_onehot, validation_data=(X_validation, y_validation_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            # Evaluate the performance of the trained model based on the validation data
            # Here we load the Trained weights from the file saved in the hard
            # disk, which should be the same as the weights of the current model.
            model.load_weights(filepath)
            y_pred = model.predict(X_validation).argmax(axis=-1)
            labels = y_validation_onehot.argmax(axis=-1)
            acc[sub, train_idx] = accuracy_score(labels, y_pred)
            kappa[sub, train_idx] = cohen_kappa_score(labels, y_pred)

            # Get the current 'OUT' time to calculate the 'run' training time
            EndTrainIdx = time.time()
            # Print & write performance measures for each run
            info = 'Subject: {}   Train no. {}   time: {:.1f} m   '.format(
                sub + 1, train_idx + 1, ((EndTrainIdx - StartTrainIdx) / 60))
            info = info + 'Validate_acc: {:.4f}   Test_kappa: {:.4f}'.format(
                acc[sub, train_idx], kappa[sub, train_idx])

            print(info)
            log_write.write(info + '\n')

            # If current training run is better than previous runs, save the history.
            if BestSubjAcc < acc[sub, train_idx]:
                BestSubjAcc = acc[sub, train_idx]
                BestTrainingHistory = history

        # Store the path of the best model among several runs
        best_run = np.argmax(acc[sub, :])
        filepath = '/saved models/subject_{}/run_{}.keras'.format(sub + 1, best_run + 1) + '\n'
        best_models.write(filepath)

        # Get the current 'OUT' time to calculate the subject training time
        EndSub = time.time()

        # Print & write the best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(
            sub + 1, best_run + 1, ((EndSub - StartSub) / 60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(
            acc[sub, best_run], np.average(acc[sub, :]), acc[sub, :].std())
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
            kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub, :].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info + '\n')

        # Plot Learning curves
        if LearnCurves:
            print('Plot Learning Curves ....... ')
            draw_learning_curves(BestTrainingHistory, sub + 1)

    # Get the current 'OUT' time to calculate the overall training time
    EndTrain = time.time()
    info = '\nTime: {:.1f} h   '.format((EndTrain - StartTrain) / (60 * 60))
    print(info)
    log_write.write(info + '\n')

    # Close open files
    best_models.close()
    log_write.close()


# %% Evaluation
def test(model, dataset_conf, results_path):
    # Open the "Log" file to write the evaluation results.
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best_models.txt", "r")

    # Get dataset parameters.
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_subjects')
    isStandard = dataset_conf.get('standardize')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('labels_name')

    # Initialize variables
    acc_BestRun = np.zeros(n_sub)
    kappa_BestRun = np.zeros(n_sub)
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    # Iteration over subjects
    # for sub in range(n_sub-1, n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    # inference_time: classification time for one trial
    # inference_time = 0
    for sub in range(n_sub):  # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(
            DataSet=dataset,
            SubjectIndex=sub + 1,
            LOSO=LOSO,
            Standardization=isStandard,
            RandomState=sub + 1
        )
        filepath = best_models.readline()
        # Load the model of the seed.
        model.load_weights(results_path + filepath[:-1])
        # Predict
        # inference_time = time.time()
        y_pred = model.predict(X_test).argmax(axis=-1)
        # inference_time = (time.time() - inference_time) / X_test.shape[0]

        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        acc_BestRun[sub] = accuracy_score(labels, y_pred)
        kappa_BestRun[sub] = cohen_kappa_score(labels, y_pred)

        # Calculate and draw confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='true')
        draw_confusion_matrix(cf_matrix[sub, :, :], str(sub + 1), results_path, classes_labels)

        # Print & write performance measures for each subject
        best_run = filepath[filepath.find('run_') + 4:filepath.find('.keras')]
        info = 'Subject: {}   best_run: {:2}  '.format(sub + 1, best_run)
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_BestRun[sub], kappa_BestRun[sub])
        log_write.write('\n' + info)
        print(info)

    # Print & write the average performance measures for all subjects
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_BestRun), np.average(kappa_BestRun)
    )
    print(info)
    log_write.write('\n' + info)

    # Draw a performance bar chart for all subjects
    draw_performance_barChart(n_sub, acc_BestRun, 'Accuracy')
    draw_performance_barChart(n_sub, kappa_BestRun, 'k-score')
    # Draw confusion matrix for all subjects (average)
    draw_confusion_matrix(cf_matrix.mean(0), 'All Subjects', results_path, classes_labels)

    # Close opened file
    log_write.close()


# %% Running this procedure
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
    results_path = os.getcwd() + "/PretrainResults"
    if not os.path.exists(results_path):
        # Create a new directory if it does not exist.
        os.makedirs(results_path)

    # Set dataset parameters.
    dataset_conf = {
        "name": DataSet,
        "n_classes": NumClasses,
        "labels_name": LabelsName,
        "n_subjects": NumSubjects,
        "n_channels": NumChannels,
        "n_timepoints": SamplingFrequency,
        "standardize": False,
        "LOSO": False,
        "data_filtering": False,
        "drop_non-target_sample": True,
        "data_augmentation": False,
        'augment_epoch': 5000,
        'augment_rate': 0.1,
        'remain_rate': 1
    }

    # Set training hyperparameters。
    train_conf = {
        "batch_size": 64,
        "epochs": 1000,
        "patience": 300,
        "learning_rate": 0.0009,
        "n_trains": 10,
        "LearnCurves": True,
        "model_name": "EEG_DBNet_V2",
        "GPU_idx": 0
    }

    # Choose GPU
    set_gpu_device(train_conf.get("GPU_idx"))

    # Train the model
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the "/results" folder.
    model = get_model(train_conf.get("model_name"), dataset_conf)
    test(model, dataset_conf, results_path)


# %%
if __name__ == "__main__":
    run()
