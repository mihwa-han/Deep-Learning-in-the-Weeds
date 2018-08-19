## Import necessary packages
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from os.path import exists, expanduser
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir
from os.path import isfile, join

from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.optimizers import Adam
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.datasets import load_files


class seedlings:

    '''
    seedlings class: wrap the data pre-processing, model training, and validation
    steps into a single class object.

    Input variables:
        - path (): the full file path to the working directory
        - path_training (default='./train'): the path from working directory to the directory with training data
        - path_test (default = './test'): the path from working directory to the directory with the test data
        - use_model3 (default=False): use model3 instead of default model1
        - use_xception (default=False): use xception model instead of default model1
        - wt_file (default=): pre-trained weights file (for transfer learning)
        - n_epochs (default=50): number of epochs to train model
        - btch_size (default=32): batch size to use for training model
        - aug (default=False): whether to use augmentation to supplement training set images
        - ave (default='weighted'): type of averaging to use in training model
        - img_size (default=47): size to rescale input images to (length in pixels)
        - rnd_state (default=42): random state integer used to make train/test split
        - make_plot (default=False) : choose whether to make a confusion matrix plot or not
        - pltname (default='conf_matrix.png') : name of confusion matrix plot

    '''
    def __init__(self, path='.', path_training='./train', path_test='./test', use_model3=False,
    use_xception=False, wt_file=None, n_epochs=5, btch_size=20, aug=False, ave='weighted', img_size=47,
    rnd_state=42, make_plot=False, pltname='conf_matrix.png'):

        self.path=path
        self.path_training=path_training
        self.path_test=path_test
        self.use_model3=use_model3
        self.use_xception=use_xception
        self.wt_file=wt_file
        self.n_epochs=n_epochs
        self.btch_size=btch_size
        self.aug=aug
        self.ave=ave
        self.img_size=img_size
        self.rnd_state = rnd_state

        ## 12 seedlings
        categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
                    'Common wheat', 'Fat Hen', 'Loose Silky-bent',
                    'Maize', 'Scentless Mayweed', 'Shepherds Purse',
                    'Small-flowered Cranesbill', 'Sugar beet']

        abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']

        # Create train/test/validation samples
        train_tensors, valid_tensors, test_tensors, y_train, y_valid, y_test, labels = self.train_valid_test_split()
        # Use model1 by default, but if model3 or xception are selected then use those instead
        if not use_model3:
            if not use_xception:
                model = self.model1(train_tensors)
                if not self.wt_file:
                    self.wt_file = 'weights.model1_rmsprop_with_aug.hdf5'
            else:
                model=self.xception()
                if not self.wt_file:
                    self.wt_file = 'weights.Xception_with_aug.hdf5'
        else:
            model = self.model3(train_tensors)
            if not self.wt_file:
                self.wt_file = 'weights.model1_rmsprop_with_aug.hdf5'


        # train model
        #### ADD HERE: check whether the weight file and training data exists to catch possible errors!
        y_pred, predictions,test_list = self.train_model(model, RMSprop(), train_tensors, valid_tensors, test_tensors, y_train, y_valid, y_test, labels)
        if make_plot:
            self.plot_confusion(pltname,categories,abbreviation,predictions,test_list)

    def plot_confusion(self, pltname,categories,abbreviation,predictions,test_list):
        ## Check a confusion matrix with the result using test dataset.
        confusion = confusion_matrix(test_list,predictions)
        df = pd.DataFrame({'class': categories, 'abbreviation': abbreviation})

        ## Plot Confusion Matrix
        fig, ax = plt.subplots(1)
        ax = sns.heatmap(confusion, ax=ax, cmap=plt.cm.Oranges, annot=True)
        ax.set_xticklabels(abbreviation)
        ax.set_yticklabels(abbreviation)
        plt.title('Confusion Matrix',size=20)
        plt.ylabel('True',size=16)
        plt.xlabel('Predicted',size=16)
        plt.savefig(pltname,dpi=150)
        '''

        ## Apply the model to the test file and save the result for kaggle posting
        model1.load_weights(wt_file)
        data=load_files('./test')
        final_X_test = np.array(data['filenames'])
        final_test_tensors = imgs_to_tensor('./test/'+df_test.file.values,47).astype('float32')/255

        predictions = [np.argmax(model1.predict(np.expand_dims(feature, axis=0)))
                     for feature in final_test_tensors]
        y_pred = [labels[i] for i in predictions]

        df = pd.DataFrame(data={'file': df_test['file'], 'species': y_pred})
        df_sort = df.sort_values(by=['file'])
        df_sort.to_csv('final.csv', index=False)

        files, targets = self.load_dataset(path)

        '''


    ## Preprocessing 1 : load files and convert the labels for each seeding to binary class matrix
    def load_dataset(self,path):
        data = load_files(path)
        files = np.array(data['filenames'])
        targets = np_utils.to_categorical(np.array(data['target']), 12)
        return files, targets

    ## Preprocessing 2: reshape all input images to the appropriate tensors with dimension, (Number, 3, width, height)
    def img_to_tensor(self, img_path):
        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)

    def imgs_to_tensor(self, img_paths):
        list_of_tensors = [self.img_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)


    # Perform model training
    def train_model(self, model, opt, train_tensors, valid_tensors, test_tensors, y_train, y_valid, y_test, labels):
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        ## save the model weights using ModelCheckpoint
        checkpointer = [EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                          ModelCheckpoint(filepath=self.wt_file, monitor='val_loss',
                              save_best_only=True, verbose=0),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0,
                                mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]


        ## train the model with train dataset and validation dataset

        # Do image augmentation if requested
        if self.aug:
          datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

          model.fit_generator(datagen.flow(train_tensors, y_train, batch_size=self.btch_size),
                      steps_per_epoch=len(train_tensors)/self.btch_size,
                      validation_data=datagen.flow(valid_tensors, y_valid, batch_size=self.btch_size),
                      validation_steps=len(valid_tensors)/self.btch_size,
                      callbacks=checkpointer,
                      epochs=self.n_epochs,
                      verbose=1)

        else:
          model.fit(train_tensors, y_train, validation_data=(valid_tensors, y_valid),
                    epochs=self.n_epochs, batch_size=self.btch_size, callbacks=checkpointer, verbose=1)

        ## load the weight and make a prediction using test data set.
        model.load_weights(self.wt_file)
        predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0)))
                     for feature in test_tensors]
        y_pred = [labels[i] for i in predictions]
        test_list = y_test.argmax(axis=1)

        print(f1_score(test_list, predictions, average=self.ave))
        print(accuracy_score(test_list, predictions))

        return(y_pred, predictions,test_list)


    ## train(80%)/validation(10%)/test(10%) split
    def train_valid_test_split(self):

        labels = listdir(self.path_training)
        train_files, train_targets = self.load_dataset(self.path_training)

        y_train = train_targets
        train_tensors = self.imgs_to_tensor(train_files).astype('float32')/255

        strati = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.rnd_state)
        for train_index, test_index in strati.split(train_tensors, y_train):
            train_tensors, valid_tensors = train_tensors[train_index], train_tensors[test_index]
            y_train, y_valid = y_train[train_index], y_train[test_index]

        strati = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=self.rnd_state)
        for train_index, test_index in strati.split(valid_tensors, y_valid):
            valid_tensors, test_tensors = valid_tensors[train_index], valid_tensors[test_index]
            y_valid, y_test = y_valid[train_index], y_valid[test_index]

        return(train_tensors, valid_tensors, test_tensors, y_train, y_valid, y_test,labels)


    ## Build Model 1
    def model1(self,train_tensors):
        model = Sequential()
        model.add(Conv2D(filters=32,kernel_size=2, activation='relu',
                       input_shape=train_tensors.shape[1:]))
        model.add(Conv2D(filters=32,kernel_size=2, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
        model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=128,kernel_size=2, activation='relu'))
        model.add(Conv2D(filters=128,kernel_size=2, activation='relu'))
        model.add(MaxPooling2D())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024,activation='relu'))
        model.add(Dense(1024,activation='relu'))
        model.add(Dense(12,activation='softmax'))

        return model


    ## Build Model 3
    def model3(self,train_tensors):
        model = Sequential()
        model.add(Conv2D(filters=64,kernel_size=2, activation='relu',
                       input_shape=train_tensors.shape[1:]))
        model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=128,kernel_size=2, activation='relu'))
        model.add(Conv2D(filters=128,kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
        model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.2))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(12,activation='softmax'))

        return model

    # Build transfer learning model  xception model
    def xception(self):
        ## load Xception model from keras package
        pre_train = Xception(input_shape=(128,128, 3), include_top=False, weights='imagenet', pooling='avg')
        ## complete the model with Xception with a fully connected layer (and dropout)
        x = pre_train.output
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(12, activation='softmax')(x)
        model = Model(inputs=pre_train.input, outputs=predictions)

        return model
