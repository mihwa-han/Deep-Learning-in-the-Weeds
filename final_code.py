## Python Package
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
                 
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

## 12 seedlings
categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 
              'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 
              'Small-flowered Cranesbill', 'Sugar beet']

## Preprocessing 1 : load files and convert the labels for each seeding to binary class matrix
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 12)
    return files, targets

## Preprocessing 2: reshape all input images to the appropriate tensors with dimension, (Number, 3, width, height) 
def img_to_tensor(img_path,size):
    img = image.load_img(img_path, target_size=(size,size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def imgs_to_tensor(img_paths,size):
    list_of_tensors = [img_to_tensor(img_path,size) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)



## train(80%)/validation(10%)/test(10%) split 
def train_valid_test_split():
    
    labels = listdir("./train")
    train_files, train_targets = load_dataset('./train')

    y_train = train_targets
    train_tensors = imgs_to_tensor(train_files,47).astype('float32')/255

    strati = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
    for train_index, test_index in strati.split(train_tensors, y_train):
        train_tensors, valid_tensors = train_tensors[train_index], train_tensors[test_index]
        y_train, y_valid = y_train[train_index], y_train[test_index]

    strati = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42) 
    for train_index, test_index in strati.split(valid_tensors, y_valid):
        valid_tensors, test_tensors = valid_tensors[train_index], valid_tensors[test_index]
        y_valid, y_test = y_valid[train_index], y_valid[test_index]
    
    return(train_tensors,valid_tensors,test_tensors,y_train,y_valid,y_test)

train_tensors,valid_tensors,test_tensors,y_train,y_valid,y_test = train_valid_test_split()

## Model 1
## Build Model 1, adding each layer 
def model1():
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
    
    return(model)

## Model 3
## Build Model 3, adding each layer
def model3():
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

    return(model)

def xception():
    ## load Xception model from keras package
    pre_train = Xception(input_shape=(128,128, 3), include_top=False, weights='imagenet', pooling='avg')
    ## complete the model with Xception with a fully connected layer (and dropout)
    x = pre_train.output
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(12, activation='softmax')(x)
    model_Xception = Model(inputs=pre_train.input, outputs=predictions)
    
    return(model_Xception)


## train_model
def train_model(model,filename,epochs,batch_size,augmentation=False,opt,ave):
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

    ## save the model weight using ModelCheckpoint
    checkpointer = [EarlyStopping(monitor='val_loss', patience=5, verbose=0), 
            ModelCheckpoint(filepath=filename, 
                            monitor='val_loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, 
                              mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]

    ## train the model1 with train dataset and validation dataset
    if augmentation:
        datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True)

        model1.fit_generator(datagen.flow(train_tensors, y_train, batch_size=batch_size),
                    steps_per_epoch=len(train_tensors)/batch_size, 
                    validation_data=datagen.flow(valid_tensors, y_valid, batch_size=batch_size), 
                    validation_steps=len(valid_tensors)/batch_size,
                    callbacks=checkpointer,
                    epochs=epochs, 
                    verbose=1)

    else:
        model.fit(train_tensors, y_train, 
                  validation_data=(valid_tensors, y_valid),
                  epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

    ## load the weight and make a prediction using test data set.
    model.load_weights(filename)
    predictions = [np.argmax(model1.predict(np.expand_dims(feature, axis=0))) 
                   for feature in test_tensors]
    y_pred = [labels[i] for i in predictions]
    test_list = y_test.argmax(axis=1)

    print(f1_score(test_list, predictions, average=ave)) 
    print(accuracy_score(test_list,predictions))
    
    return(y_pred,test_list)


def main():
    model1 = model1()
    #model3 = model3()
    #model_Xception = xception()

    #y_pred,test_list = train_model(model1,'weights.model1_rmsprop.hdf5',20,20,RMSprop(),'macro')
    y_pred,test_list = train_model(model1,'weights.model1_rmsprop_with_aug.hdf5',50,32,False,RMSprop(),'weighted')
    #y_pred,test_list = train_model(model_Xception,'weights.Xception_with_aug.hdf5',5,32,True,RMSprop(),'weighted')


    ## Check a confusion matrix with the result using test dataset.
    confusion = confusion_matrix(test_list,predictions)
    abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']
    pd.DataFrame({'class': categories, 'abbreviation': abbreviation})

    ## Plot Confusion Matrix
    # import seaborn as sns
    # fig, ax = plt.subplots(1)
    # ax = sns.heatmap(confusion, ax=ax, cmap=plt.cm.Oranges, annot=True)
    # ax.set_xticklabels(abbreviation)
    # ax.set_yticklabels(abbreviation)
    # plt.title('Confusion Matrix',size=20)
    # plt.ylabel('True',size=16)
    # plt.xlabel('Predicted',size=16)
    # plt.show();

    ## Apply the model to the test file and save the result for kaggle posting
    model1.load_weights('weights.model1_rmsprop_with_aug.hdf5')
    data=load_files('./test')
    final_X_test = np.array(data['filenames'])
    final_test_tensors = imgs_to_tensor('./test/'+df_test.file.values,47).astype('float32')/255

    predictions = [np.argmax(model1.predict(np.expand_dims(feature, axis=0))) 
                   for feature in final_test_tensors]
    y_pred = [labels[i] for i in predictions]

    df = pd.DataFrame(data={'file': df_test['file'], 'species': y_pred})
    df_sort = df.sort_values(by=['file'])
    df_sort.to_csv('final.csv', index=False)
    
if __name__=="__main__":
    main()

