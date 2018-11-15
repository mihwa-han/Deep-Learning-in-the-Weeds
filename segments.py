import numpy as np
import cv2

def create_mask_for_plant(image):
    '''
    Take an input image, and produce a mask that consists of the same pixel 
    dimensions as the input image, but at the position of pixels that fall 
    in a range of color shades we are interested in there will be a value 
    of "1", and at all other pixels there will be a "0" value.
    :param: image: data for input image
    :returns: mask: corresponding mask image
    '''
    # Convert input image from RGB to HSV (Hue, Saturation, Value)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of HSV values to allow to pass through our filter. 
    # Note that the sensitivity gives us some flexibility in the range 
    # of hues to include, and the higher the sensitivity the weaker 
    # the resulting filter will be
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    # Produce a mask image
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    return mask

def segment_plant(image):
    '''
    Make a mask image from an input image (using the previous function), and 
    create a segmented plant image (an image where we have only the pixels 
    consisting of the plants as nonzero).
    :param: image: data for input image
    :returns: output: segmented plant image
    '''
    # Create mask image
    mask = create_mask_for_plant(image)
    
    # Produce segmented image by performing a bitwise AND operation on the 
    # input image and the mask image, thereby setting all pixels that are 
    # NOT == 1 in the mask to zero in the output image
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def make_segmented_images(filepath, img_size):
    '''
    End-to-end function to read in all images in a directory, resize them to 
    an appropriate dimention, and create segmented images from them.
    :param: filepath: path to the directory with input images
    :param: img_size: no. of pixels on a side to resize the images to match
    :returns: img (resized input image), image_mask (mask image), image_segmented (segmented image)
    '''
    
    # Read in the input images and resize them
    img = cv2.imread(os.path.join('../input', filepath), cv2.IMREAD_COLOR)
    img = cv2.resize(img.copy(), img_size, interpolation = cv2.INTER_AREA)

    # Call the previous functions to make a mask image and a segmented image
    image_mask = create_mask_for_plant(img)
    image_segmented = segment_plant(img)
  
    return img, image_mask, image_segmented