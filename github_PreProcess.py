import os

import GANs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import scipy.signal as signal

from scipy import io
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical

from github_Utils import get_GANs


# https://www.kaggle.com/datasets/rramele/p300samplingdataset/code
# https://github.com/chocochip13/EEG-P300/blob/master/notebooks/EDA.ipynb

# %%
def load_data_P300_ITBA(DataPath, SubjectIndex, StartTime=0, EndTime=1, SamplingFrequency=250, DropStimulations=4):
    """
    Loading and dividing of the dataset (Produced by the CiC, ITBA University, Buenos Aires, Argentina) based on the
    subject-specific (subject-dependent) approach. The original data processing method is on:
    https://www.kaggle.com/datasets/rramele/p300samplingdataset/code
    https://github.com/chocochip13/EEG-P300/blob/master/notebooks/EDA.ipynb

    :param DataPath: String
        Dataset P300 speller ITBA can be downloaded on:
        https://www.kaggle.com/datasets/rramele/p300samplingdataset/data.
    :param SubjectIndex: Int
        Number of the subject in [1, ..., 8]
    :param StartTime: Int
        Starting points of the data sequence. (Second)
    :param EndTime:
        Ending points of the data sequence. (Second)
    :param SamplingFrequency: Int
        Sampling frequency.
    :param DropStimulations: Int
        Number of stimulations to drop from the dataset, as whose length is less than sequence setting.
    """

    OriginalData = io.loadmat(DataPath + "P300S0" + str(SubjectIndex) + ".mat")['data'][0][0]

    # EEG Matrix (8 channels), (358372, 8)
    EEGData = OriginalData['X']
    # EEGChannelNames = OriginalData['channelNames'] 10-20
    # Sample point where each flashing starts
    # (sample point id, duration, stimulation points, target/non-target), (4200, 4)
    FlashInformation = OriginalData['flash']

    StartPoint = int(StartTime * SamplingFrequency)  # 0
    EndPoint = int(EndTime * SamplingFrequency)  # 256

    # (4200, 250, 8)
    X = np.zeros((len(FlashInformation) - DropStimulations, int(EndPoint - StartPoint), EEGData.shape[1]))
    y = np.zeros((len(FlashInformation) - DropStimulations))  # 4200

    for i in range(len(FlashInformation) - DropStimulations):
        event = FlashInformation[i][0]  # Onset point of an event.
        X[i, :, :] = EEGData[event + StartPoint:event + EndPoint, :]
        y[i] = FlashInformation[i][3] - 1
    X = X.transpose(0, 2, 1)

    return X, y


# %%
def load_data_P300_ALS(DataPath, SubjectIndex, StartTime=0, EndTime=1, SamplingFrequency=256):
    """
    Loading and dividing of the dataset (Produced by the BCI2000) based on the subject-specific
    (subject-dependent) approach.
    https://lampx.tugraz.at/~bci/database/008-2014/description.pdf

    :param DataPath: String
        Dataset P300 speller ALS can be downloaded on:
        https://bnci-horizon-2020.eu/database/data-sets
    :param SubjectIndex: Int
        Number of the subject in [1, ..., 8]
    :param StartTime: Int
        Starting points of the data sequence. (Second)
    :param EndTime:
        Ending points of the data sequence. (Second)
    :param SamplingFrequency: Int
        Sampling frequency.
    """

    OriginalData = io.loadmat(DataPath + "A0" + str(SubjectIndex) + ".mat")['data'][0, 0]

    # Channel's name (10-10 system).
    # EEGChannelNames = OriginalData[0]
    # EEG data (347704, 8).
    EEGData = OriginalData[1]
    # EEG data (347704).
    EEGLabel = OriginalData[2]
    # Trail onset.
    TrailOnset = OriginalData[4][0]

    DataSequence = (EndTime - StartTime) * SamplingFrequency

    X = []
    y = []

    for trail_idx in range(len(TrailOnset)):
        TrailStartPoint = TrailOnset[trail_idx]
        if trail_idx < len(TrailOnset) - 1:
            TrailEndPoint = TrailOnset[trail_idx + 1]
        else:
            TrailEndPoint = len(EEGLabel)

        TempleLabel = EEGLabel[TrailStartPoint:TrailEndPoint]

        for label_idx in range(1, len(TempleLabel)):
            if TempleLabel[label_idx] != TempleLabel[label_idx - 1]:
                if TempleLabel[label_idx] == 1:
                    label = 0
                elif TempleLabel[label_idx] == 2:
                    label = 1
                else:
                    continue

                # Extract the sequence of 256 points.
                data = EEGData[TrailStartPoint + label_idx: TrailStartPoint + label_idx + DataSequence, :]

                # Ensure the sequence is of length 256.
                if len(data) == DataSequence:
                    X.append(data)
                    y.append(label)
                else:
                    # Exit if the sequence is shorter than 256.
                    break

    # Convert lists to numpy arrays
    X = np.array(X)
    X = X.transpose(0, 2, 1)
    y = np.array(y)

    return X, y


# %%
def load_data_loso(DataPath, SubjectIndex, DataSet):
    """
    Loading and dividing of the data set based on the 'Leave One Subject Out (LOSO)' evaluation approach.

    LOSO is used for subject-independent evaluation. In LOSO, the model is trained and evaluated by several folds,
    equal to the number of subjects, and for each fold, one subject is used for evaluation and the others for
    training.

    The LOSO evaluation technique ensures that separate subjects (not visible in the training data) are used to
    evaluate the model.

    :param DataPath: String
        Path to the data folder.
    :param SubjectIndex: Int
        Number of a subject in [1, 2, ..., 8].
        The subject data is used to test the model and other subject data for training
    :param DataSet: String
        Chose the dataset.
    :return: Data and labels.
    """

    X_train, y_train, X_sub, y_sub, X_test, y_test = None, None, None, None, None, None

    for sub in range(1, 9):
        if DataSet == "P300_ALS":
            X_sub, y_sub = load_data_P300_ALS(DataPath, sub)
        elif DataSet == "P300_ITBA":
            X_sub, y_sub = load_data_P300_ITBA(DataPath, sub)

        if sub == SubjectIndex:
            X_test = X_sub
            y_test = y_sub
        # elif not X_train.any():
        # elif not X_train.all():
        elif X_train is None:
            X_train = X_sub
            y_train = y_sub
        else:
            X_train = np.concatenate((X_train, X_sub), axis=0)
            y_train = np.concatenate((y_train, y_sub), axis=0)

    return X_train, y_train, X_test, y_test


# %%
def standardize_data(X_train, X_test, channels, scale_type='Standard'):
    """
    By using the statistical values from the training set for standardization, it ensures that the normalization of
    the test data is based on the distribution of the training data. This is a common practice to ensure that the
    model evaluation is conducted on a data distribution similar to the training data.
    """

    print("\n*********************************************************************")
    print("Data will be scaled by '{}' method.".format(scale_type))
    X_train_s = np.zeros(X_train.shape)
    X_test_s = np.zeros(X_test.shape)

    for j in range(channels):
        if scale_type == 'Standard':
            scaler = StandardScaler()
        elif scale_type == 'OneMinusOne':
            scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise Exception("'{}' dataset is not supported yet!".format(scale_type))
        X_train_s[:, j, :] = scaler.fit_transform(X_train[:, j, :])
        X_test_s[:, j, :] = scaler.transform(X_test[:, j, :])

    return X_train_s, X_test_s


# %%
def bandpass_filter(data, FrequencyBand, SamplingFrequency, FilterOrder=50, FilteringAxis=-1, FilteringType='lfilter'):
    """
    Bandpass filtering.

    :param data: Numpy array
        Data to be filtered.
    :param FrequencyBand: Tuple.
        Filter frequency cutoff.
    :param SamplingFrequency: list
        Sampling frequency.
    :param FilterOrder: Int
        Order of the Filter.
    :param FilteringAxis: Int
        Which axis to be filtered.
    :param FilteringType:
        Which filtering type to be used.
    :return: Numpy array
    """
    a = [1]

    if (FrequencyBand[0] == 0 or FrequencyBand[0] is None) and (FrequencyBand[1] is None or
                                                                FrequencyBand[1] == SamplingFrequency / 2.0):
        # no filter
        print("Not doing any filtering. Invalid cut-off specifications")
        return data
    elif FrequencyBand[0] == 0 or FrequencyBand[0] is None:
        # low-pass filter
        print("Using lowpass filter since low cut hz is 0 or None")
        h = signal.firwin(numtaps=FilterOrder + 1, cutoff=FrequencyBand[1], pass_zero="lowpass", fs=SamplingFrequency)
    elif (FrequencyBand[1] is None) or (FrequencyBand[1] == SamplingFrequency / 2.0):
        # high-pass filter
        print("Using highpass filter since high cut hz is None or nyquist freq")
        h = signal.firwin(numtaps=FilterOrder + 1, cutoff=FrequencyBand[0], pass_zero="highpass", fs=SamplingFrequency)
    else:
        h = signal.firwin(numtaps=FilterOrder + 1, cutoff=FrequencyBand, pass_zero="bandpass", fs=SamplingFrequency)

    if FilteringType == 'filtfilt':
        FiltData = signal.filtfilt(h, a, data, axis=FilteringAxis)
    else:
        FiltData = signal.lfilter(h, a, data, axis=FilteringAxis)

    return FiltData


# %%
def get_data(DataSet, SubjectIndex, RandomState, LOSO=False, Standardization=False, DataFiltering=False,
             DropNontargetSample=False, DataAugment=False, AugmentRate=0.5, NumAugEpochs=1000, RemainRate=1):
    # Load and split the dataset into training and testing
    X_train, X_test, y_train, y_test = None, None, None, None
    if DataSet == "P300_ITBA":
        DataPath = "/data4/louxicheng/EEG_data/Visual_Evoked_Potentials/P300_speller_ITBA/"
        SamplingFrequency = 256
        if LOSO:
            # Loading and Dividing of the data set based on the 'Leave One Subject Out' (LOSO) evaluation approach.
            X_train, y_train, X_test, y_test = load_data_loso(DataPath, SubjectIndex, DataSet=DataSet)
        else:
            # Loading and Dividing of the data set based on the subject-specific (subject-dependent) approach.
            X, y = load_data_P300_ITBA(
                DataPath,
                SubjectIndex,
                SamplingFrequency=SamplingFrequency
            )
            train_test_data_split = StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=RandomState
            )
            for train_index, test_index in train_test_data_split.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

    elif DataSet == "P300_ALS":
        DataPath = "/data4/louxicheng/EEG_data/Visual_Evoked_Potentials/P300_speller_ALS/"
        SamplingFrequency = 256
        if LOSO:
            # Loading and Dividing of the data set based on the 'Leave One Subject Out' (LOSO) evaluation approach.
            X_train, y_train, X_test, y_test = load_data_loso(DataPath, SubjectIndex, DataSet=DataSet)
        else:
            # Loading and Dividing of the data set based on the subject-specific (subject-dependent) approach.
            X, y = load_data_P300_ALS(
                DataPath,
                SubjectIndex,
                SamplingFrequency=SamplingFrequency
            )
            train_test_data_split = StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=RandomState
            )
            for train_index, test_index in train_test_data_split.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

    else:
        raise Exception("'{}' dataset is not supported yet!".format(DataSet))

    # Standardize the data
    if Standardization:
        n_trails, n_channels, n_timepoints = X_train.shape
        X_train, X_test = standardize_data(X_train, X_test, n_channels)

    # data augment
    if DataAugment:
        target_data_augment = get_GANs(
            X_train=X_train,
            y_train=y_train,
            SubjectIndex=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate,
            DataAugment=DataAugment
        )

        X_train = np.concatenate((X_train, target_data_augment), axis=0)
        y_train = np.concatenate((y_train, np.ones(target_data_augment.shape[0])), axis=0)

    if DropNontargetSample:
        X_reshaped = X_train.reshape((X_train.shape[0], -1))
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=RandomState)
        X_undersample, y_train = undersample.fit_resample(X_reshaped, y_train)

        # Save the non-target samples in the pretraining.
        pretrain_nontarget_samples = X_undersample[y_train == 0]
        np.save('PretrainResults/pretrain_nontarget_samples.npy', pretrain_nontarget_samples)

        X_train = X_undersample.reshape(
            (X_undersample.shape[0], X_train.shape[1], X_train.shape[-1])
        )
    else:
        if not DataAugment:
            remain_nontarget_data_idx = X_train[y_train == 0].shape[0] - X_train[y_train == 1].shape[0]
            remain_nontarget_data = int(remain_nontarget_data_idx * RemainRate)

            target_data_count = X_train[y_train == 1].shape[0]

            remain_nontarget_data = int(remain_nontarget_data)
            nontarget_data_to_keep = target_data_count + remain_nontarget_data

            X_train_nontarget_subset = X_train[y_train == 0][:nontarget_data_to_keep]
            y_train_nontarget_subset = y_train[y_train == 0][:nontarget_data_to_keep]

            X_train = np.concatenate((X_train_nontarget_subset, X_train[y_train == 1]), axis=0)
            y_train = np.concatenate((y_train_nontarget_subset, y_train[y_train == 1]), axis=0)

        else:
            raise Exception("DataAugment should not be applied if you want to remain reduntant non-target data.")

    # Prepare training data
    n_trails, n_channels, n_timepoints = X_train.shape
    X_train = X_train.reshape(n_trails, n_channels, n_timepoints, 1)
    y_train_onehot = to_categorical(y_train)

    # Prepare testing data
    n_trails, n_channels, n_timepoints = X_test.shape
    X_test = X_test.reshape(n_trails, n_channels, n_timepoints, 1)
    y_test_onehot = to_categorical(y_test)

    # Frequency filter
    if DataFiltering:
        FilterBanks = [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
        X_train_temp = np.zeros(X_train.shape + (len(FilterBanks),))
        X_test_temp = np.zeros(X_test.shape + (len(FilterBanks),))
        for i in range(len(FilterBanks)):
            X_train_temp[:, :, :, :, i] = bandpass_filter(data=X_train, FrequencyBand=FilterBanks[i],
                                                          SamplingFrequency=SamplingFrequency, FilteringAxis=-2)
            X_test_temp[:, :, :, :, i] = bandpass_filter(data=X_test, FrequencyBand=FilterBanks[i],
                                                         SamplingFrequency=SamplingFrequency, FilteringAxis=-2)
        X_train = X_train_temp.squeeze(-2)
        X_test = X_test_temp.squeeze(-2)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


# %%
def get_data_finetune(DataSet, SubjectIndex, RandomState, AugmentRate=0.1, NumAugEpochs=2000, GANsModel='EEG_GANet',
                      ClassifyModel='EEG_DBNet_V2', TrainGANs=False):
    # Load and split the dataset into training and testing
    X_train, X_test, y_train, y_test = None, None, None, None
    if DataSet == "P300_ITBA":
        DataPath = "/data4/louxicheng/EEG_data/Visual_Evoked_Potentials/P300_speller_ITBA/"
        SamplingFrequency = 256
        X, y = load_data_P300_ITBA(
            DataPath,
            SubjectIndex,
            SamplingFrequency=SamplingFrequency
        )
    elif DataSet == "P300_ALS":
        DataPath = "/data4/louxicheng/EEG_data/Visual_Evoked_Potentials/P300_speller_ALS/"
        SamplingFrequency = 256
        X, y = load_data_P300_ALS(
            DataPath,
            SubjectIndex,
            SamplingFrequency=SamplingFrequency
        )
    else:
        raise Exception("'{}' dataset is not supported yet!".format(DataSet))

    train_test_data_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RandomState
    )
    for train_index, test_index in train_test_data_split.split(X, y):
        X_train, _ = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]

    if TrainGANs:
        X_train_target = get_GANs(
            X_train=X_train,
            y_train=y_train,
            SubjectIndex=SubjectIndex,
            SamplingFrequency=SamplingFrequency,
            NumAugEpochs=NumAugEpochs,
            AugmentRate=AugmentRate,
            DataAugment=GANsModel
        )
    else:
        model = GANs.GAN_generator_1(SamplingFrequency=SamplingFrequency, NumChannels=8).build_model()
        model.load_weights(
            '../Paper_2/'+GANsModel+'/'+GANsModel+'_generator_AB_subject_' + str(SubjectIndex) + '.keras'
        )
        GAN_model = model
        real_target_data = X_train[y_train == 1]
        real_nontarget_data = X_train[y_train == 0]

        valid_rates = [i / 10 for i in range(1, 11)]
        if AugmentRate not in valid_rates:
            raise ValueError(f"AugmentRate must be one of the following values: {valid_rates}")
        augment_data = int((real_nontarget_data.shape[0] - real_target_data.shape[0]) * AugmentRate)
        nontarget_idx = np.random.randint(0, real_nontarget_data.shape[0], augment_data)
        real_nontarget_data_transform = real_nontarget_data[nontarget_idx]
        X_train_target = GAN_model.predict(real_nontarget_data_transform)

    pretrain_nontarget_samples = np.load('../Paper_2/'+ClassifyModel+'/pretrain_nontarget_samples.npy')
    nontarget_samples_mask = (y_train == 0)
    not_in_pretrain_mask = np.array(
        [not any(np.array_equal(x, p) for p in pretrain_nontarget_samples) for x in X_train]
    )
    remaining_nontarget_samples = X_train[nontarget_samples_mask & not_in_pretrain_mask]
    X_train_nontarget = remaining_nontarget_samples[:len(X_train_target)]

    X_finetune = np.concatenate(([X_train_target, X_train_nontarget]), axis=0)
    y_finetune = np.concatenate(([np.ones(len(X_train_target)), np.zeros(len(X_train_nontarget))]), axis=0)
    y_finetune_onehot = to_categorical(y_finetune)

    return X_finetune, y_finetune, y_finetune_onehot
