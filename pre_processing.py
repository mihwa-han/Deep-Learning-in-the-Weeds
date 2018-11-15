from PIL import Image
from keras.preprocessing import image 
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tqdm import tqdm
import random
import numpy as np

## preprocessing - convert dimensions
def img_to_tensor(img_path, size, grey = False):
    """
    Convert input image to tensor
    :param: img_path[str]: path 
    :param: size[int]: no. of pixels on one side to resize your image to match
    :param: grey[bool]: change to grey?
    :returns: tensor: (1, size, size, channel)  
    """
    if grey:
        img = Image.open(img_path).convert('L').resize((size,size))
    else:
        img = image.load_img(img_path, target_size = (size,size))
        
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def imgs_to_tensor(img_paths, size, grey = False):
    """
    Convert input images to tensor
    :param: img_paths[list]: list of paths 
    :param: size[int]: no. of pixels on one side to resize your images to match
    :param: grey[bool]: change to grey?
    :returns: tensor: (no. of images, size, size, channel)  
    """
    list_of_tensors = [img_to_tensor(img_path, size, grey = grey) 
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def convert_to_tensors(df, size, grey):
    """
    Convert input images to tensor
    :param: idf: DataFrame of paths and labels
    :param: size[int]: no. of pixels on one side to resize your images to match
    :param: grey[bool]: change to grey?
    :returns: tensor, y 
    """
    tensors = imgs_to_tensor(df.file.values, size, grey = grey).astype('float32')/255
    y = np_utils.to_categorical(np.array(df.label.values), 12)
    return tensors, y

def sampling(df, tensors, y, sampling = 0, number = None):
    """
    Choose sampling method
    :param: df: DataFrame of paths and labels 
    :param: tensors:
    :param: y: matrix of one hot encoding labels for all input images
    :param: sampling[0, 1, 2] : 0(no change), 1(downsampling), 2(upsampling)
    :param: number[DataFrame]: no. of total images for each category
    :returns: tensor, y 
    """
    if sampling == 1: ## downsampling
        min_num = number.n.values.min()
        for i in range(12):
            sub_tensors = tensors[df.label==i][:min_num]
            sub_y = y[df.label==i][:min_num]
            if i == 0:
                final_tensors = sub_tensors
                final_y = sub_y
            else:
                final_tensors = np.vstack([final_tensors,sub_tensors])
                final_y = np.vstack([final_y,sub_y])

        tensors = final_tensors.copy()
        y = final_y.copy()
        
    elif sampling == 2: ## upsampling
        max_num = 700
        add = max_num - number.n.values

        for i in tqdm(range(12)):
            sub_df = df[df.label==i]
            for j in range(add[i]):
                rdn = random.randint(0,len(sub_df)-1)
                idx = sub_df.iloc[rdn].name
                _, sample  = image_generator(np.expand_dims(tensors[idx], axis=0), 
                                             rotation=90, h_shift=0.5)
                if (i==0) & (j==0):
                    print('start')
                    y_tmp=y[idx]
                    tensors_tmp=sample
                else:
                    tensors_tmp = np.vstack([tensors_tmp,sample])
                    y_tmp = np.vstack([y_tmp,y[idx]])

        tensors = np.vstack([tensors,tensors_tmp])
        y = np.vstack([y,y_tmp])

    else: ## unchage
        pass
    
    return tensors, y  

def specific_sampling(X_train, y_train, X_valid, y_valid, add = True):
    if add == False:
        subtract_train = 300
        subtract_valid = 35
        n = len(X_train[y_train[:,6]==1])
        n_v = len(X_valid[y_valid[:,6]==1])
        X_train = X_train[subtract_train:]
        y_train = y_train[subtract_train:]
        X_valid = X_valid[subtract_valid:]
        y_valid = y_valid[subtract_valid:]
        
    else:
        add_train = 300
        add_valid = 35

        n = len(X_train[y_train[:,0]==1])
        n_v = len(X_valid[y_valid[:,0]==1])
        for j in range(add_train):
            rdn = random.randint(0,n-1)
            _, sample  = image_generator(np.expand_dims(X_train[y_train[:,0]==1][rdn], axis=0), 
                                         rotation=90, zoom = 0.2)
            if (j==0):
                print('start')
                y_tmp=y_train[y_train[:,0]==1][rdn]
                tensors_tmp=sample
            else:
                tensors_tmp = np.vstack([tensors_tmp,sample])
                y_tmp = np.vstack([y_tmp,y_train[y_train[:,0]==1][rdn]])

        X_train = np.vstack([X_train,tensors_tmp])
        y_train = np.vstack([y_train,y_tmp])

        for j in range(add_valid):
            rdn = random.randint(0,n_v-1)
            _, sample  = image_generator(np.expand_dims(X_valid[y_valid[:,0]==1][rdn], axis=0), 
                                         rotation=90, zoom = 0.2)
            if (j==0):
                print('start')
                y_tmp=y_valid[y_valid[:,0]==1][rdn]
                tensors_tmp=sample
            else:
                tensors_tmp = np.vstack([tensors_tmp,sample])
                y_tmp = np.vstack([y_tmp,y_valid[y_valid[:,0]==1][rdn]])

        X_valid = np.vstack([X_valid,tensors_tmp])
        y_valid = np.vstack([y_valid,y_tmp])

    return X_train, y_train, X_valid, y_valid

def train_val_test(tensors, y):
    """
    train (80%) validation (10%) test (10%) split
    :param: tensors
    :param: y: matrix of one hot encoding labels for all input images
    :returns: X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(tensors, y, test_size = 0.20,
                                                        stratify = y)

    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.50,
                                                        stratify = y_test)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def image_generator(img, rotation=0., w_shift=0., h_shift=0., shear=0., 
                           zoom=0., h_flip=False, v_flip=False):
    """
    Generate new image with data augmentation options
    :params: img
    :params: rotation
    :params: w_shift, h_shift
    :params: shear
    :params: zoom
    :params: h_flip, v_flip
    :return: model, new image
    """
    datagen = ImageDataGenerator(rotation_range = rotation,
                width_shift_range = w_shift, 
                height_shift_range = h_shift,
                shear_range = shear,
                zoom_range = zoom,
                horizontal_flip = h_flip, 
                vertical_flip = v_flip,
                fill_mode = 'nearest')
    datagen.fit(img)
    return datagen, datagen.flow(img, batch_size=1)[0]

    
def image_generator_plot(datagen,img,batch_size=15):
    """
    Plot Augmented images
    :params: datagen, img
    :params: batch_size
    """
    row = int(batch_size / 7) + 1
    plt.figure(figsize=(8,row))
    for i,img_batch in enumerate(datagen.flow(img, batch_size=batch_size)):
        if i ==batch_size:
            break
        plt.subplot(row ,7,i+1)
        plt.imshow(img_batch[0])
        plt.axis('off')
    plt.show()