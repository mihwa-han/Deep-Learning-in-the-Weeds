from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score
from keras.optimizers import *
import numpy as np

def compiler(model, X_train, y_train, X_valid, y_valid, epochs, batch_size, 
             optimizer = Adam(), aug = False, verbose = 0, reduce = False,
             checkfile = 'weights.hdf5'):
    """
    Configure and fit the model 
    :param: model
    :param: X_train
    :param: y_train
    :param: X_valid
    :param: y_valid
    :param: epochs : number of epochs for training
    :param: batch_size : batch size 
    :param: optimizer : which optimizer function to use (e.g., SGD(), Adam(), RMSprop())
    :param: aug[bool] : using image augmentation
    :returns: model_info, model : loss and metrics history 
    """
    model.compile(optimizer = optimizer,
                  loss='categorical_crossentropy',
                  metrics = ['accuracy'])
    
    if reduce : 
        checkpointer = [EarlyStopping(monitor = 'val_loss', patience=10),
                        ModelCheckpoint(filepath = checkfile, 
                                       save_best_only=True),
                        ReduceLROnPlateau(monitor='val_acc', 
                            patience=3, 
                            verbose=1, 
                            factor=0.4, 
                            min_lr=0.00001)]
    else:
                checkpointer = [EarlyStopping(monitor = 'val_loss', patience=10),
                        ModelCheckpoint(filepath = checkfile, 
                                       save_best_only=True)]
    
    if aug:     
        datagen = ImageDataGenerator(rotation_range = 180,zoom_range = 0.1,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   horizontal_flip = True,vertical_flip = True)
        model_info = model.fit_generator(datagen.flow(X_train, y_train, 
                                                      batch_size = batch_size),
                                        steps_per_epoch = len(X_train)/batch_size, 
                                        validation_data = datagen.flow(X_valid, y_valid, 
                                                                     batch_size = batch_size), 
                                        validation_steps = len(X_valid)/batch_size,
                                        callbacks = checkpointer,
                                        epochs = epochs, 
                                        verbose = verbose)
    else:
        model_info = model.fit(X_train, y_train, 
                               validation_data = (X_valid, y_valid),
                               epochs = epochs, batch_size = batch_size, 
                               callbacks = checkpointer, verbose = verbose)


    return model_info, model


def evaluate(model, X_test, y_test, checkfile = 'weights.hdf5'):
    """
    Score the model with test data 
    :param: model
    :param: X_test
    :param: y_test
    :returns: test_list, predictions, f1, accuracy
    """
    model.load_weights(checkfile)
    predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) 
                   for feature in X_test]
    test_list = y_test.argmax(axis=1)
    f1 = f1_score(test_list, predictions, average = 'macro')
    accuracy = accuracy_score(test_list,predictions)   
    return test_list, predictions, f1, accuracy
