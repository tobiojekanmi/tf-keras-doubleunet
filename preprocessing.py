"""
Preprocessing Functions for Tensorflow Keras FCN Image Segmentation Models.
"""

import tensorflow as tf



################################################################################
# List image and mask filepaths function
################################################################################
def list_filepaths(image_directory, mask_directory):
    """
    Extracts image and their segmentation filenames from their
    directories.
    Args:
      image_directory: path to the image directory
      mask_directory:  path to the segmentation labels
                      directory
    Returns:
      image_paths: a list containing filepaths of the
                   images in the specified directory path
      mask_paths: a list containing filepaths of the image
                   segmentation labels in the specified
                   directory path
    """

    image_paths = []
    mask_paths = []
    image_filenames = os.listdir(image_directory)

    for image_filename in image_filenames:
        image_paths.append(image_directory + "/" + image_filename)
        mask_filename = image_filename.replace('.jpg', '.png')
        mask_paths.append(mask_directory + "/" + mask_filename)

    return image_paths, mask_paths



################################################################################
# Read Image Function
################################################################################
def read_image(image_path,
               image_size,
               image_channels=None):

    """
    Reads and resizes image from its filepath and return its equivalent  
    tensor/array representation.  
    
    Args:
        image_path: path to the image.
        image_size: a tuple containing the desired image height and width
                      respecively. i.e., (image_height, image_width).
        image_channels: (optional) number of image channels. Default to None.
    
    Returns:
        image: image tensor representation of shape 
               (image_height, image_width, image_channels)    
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. image size
    if len(image_size) != 2:
        raise ValueError('The `image_size` argument should be a tuple with image '
                         'height and width respectively i.e. (256, 256).')
    elif not (isinstance(image_size[0], int) or isinstance(image_size[1], int)):
        raise ValueError('The `image_size` argument should be a tuple containing '
                         'only integer values.')
    # 2. image channels
    if not isinstance(image_channels, int):
        raise ValueError('The `image_channels` argument should be an integer.')
    
    #--------------------------------------------------------------------------#
    # Build function
    #--------------------------------------------------------------------------#
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=image_channels,
                                  expand_animations = False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size, method='nearest')
    
    return image



################################################################################
# Read Mask Function
################################################################################
def read_mask(mask_path, 
              mask_size, 
              num_classes=None, 
              reduce_mask_dims=True):
    """
    Reads and resizes segmentation label (mask) from its filepath and return   
    its equivalent tensor/array representation.  
    Args:
        mask_path: path to the mask/segmentation label.
        image_size: a tuple containing the desired mask height and width 
                     respecively. i.e. (mask_height, mask_width). 
        num_classes: (optional) number of segmentation labels/classes in the 
                     mask.
        reduce_mask_dims: one of True (to return sparse_categorical labels i.e.,  
                          3 image segmentation labels will be in form of [1,2,3] 
                          and the last channel of the mask will be 1) or False
                          (to return categorical labels i.e., 3 image segmentation 
                          labels will be in form of [[0,0,1], [0,1,0], [1,0,0]]
                          and the last channel of the mask will equal to the 
                          number of classes or num_classes). Default to True.
    Returns:
        mask: mask/segmentation label tensor representation of shape 
               (image_height, image_width, image_channels)
    
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. image size
    if len(mask_size) != 2:
        raise ValueError('The `mask_size` argument should be a tuple with image '
                         'height and width respectively i.e. (256, 256).')
    elif not (isinstance(mask_size[0], int) or isinstance(mask_size[1], int)):
        raise ValueError('The `mask_size` argument should be a tuple containing '
                         'only integer values.')
    # 2. image channels
    if not isinstance(num_classes, int):
        raise ValueError('The `image_channels` argument should be an integer.')
    
    # 3. reduce_mask_dims
    if not isinstance(reduce_mask_dims, bool):
        raise ValueError('The `reduce_mask_dims` argument should be one of '
                         'True or False.')
        
    #--------------------------------------------------------------------------#
    # Build function
    #--------------------------------------------------------------------------#
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=image_channels, 
                                 expand_animations = False)
    if reduce_mask_dims:
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, image_size, method='nearest')
    
    return mask



################################################################################
# Data Generator Function
################################################################################
def data_generator(image_paths, 
                   mask_paths, 
                   image_size=(224,224), 
                   image_channels=None,
                   num_classes=None,
                   reduce_mask_dims=True,
                   batch_size=32, 
                   cache_dataset=True, 
                   shuffle_dataset=True, 
                   prefetch_dataset=True, 
                   preprocess_function=None):

    """
    Generates TF dataset containinng images and their segmentations.

    Arguments:
        image_paths: a list or tuple containing paths to the images.
        mask_paths: a list or tuple containing paths to the segmentations.
                    Note: The mask_paths list must follow the same format which 
                    the image_paths is (i.e. The image path in index 0 in the 
                    image_paths list must have it's segmentation/mask path to  
                    be at 0 in the mask_paths list). 
                    Inability to ensure this will lead to incorrect modelling.
        image_size: a tuple containing image height and image width respecively.
                     i.e. (image_height, image_width). Default to (224,224).
        image_channels: (optional) number of image channels.  Default to None.
        num_classes: (optional) number of segmentation labels/classes in the mask.
        reduce_mask_dims: (optional) one of True (to return sparse_categorical 
                          labels i.e., 3 image segmentation labels will be in form 
                          of [1,2,3] and the last channel of the mask will be 1) 
                          or False (to return categorical labels i.e., 3 image 
                          segmentation labels will be in form of [[0,0,1], 
                          [0,1,0], [1,0,0]] and the last channel of the mask will
                          equal to the number of classes or num_classes). Default
                          to None.
        batch_size: (optional) size of image/mask to load per batch. 
                    Default to 32.
        cache_dataset: (optional) boolean to specify whether to store dataset 
                       into memory after loading it during the first itration. 
                       Subsequent iterations will use the cached dataset. 
                       Default to True.
        shuffle_dataset: (optional) boolean to specify whether to randomly  
                         shuffle elements of the  dataset while loading them  
                         or not. Default to True.
        prefetch_dataset: (optional) boolean to specify whether to prefetch  
                         dataset or not. Default to True.
        preprocess_function: (optional) custom data preprocessing function or  
                          class. Default to None.

    Returns:
        A tensorflow.data.Dataset pipeline to load images and their 
        segmentations from the specified argument.
        """

    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. image and mask paths
    if not (isinstance(image_paths, list) or isinstance(image_paths, tuple)):
        raise ValueError('The `image_paths` argument should be a list or tuple '
                         'containing list of image filepaths.')
    if not (isinstance(mask_paths, list) or isinstance(mask_paths, tuple)):
        raise ValueError('The `mask_paths` argument should be a list or tuple '
                         'containing list of mask filepaths.')
    
    # 2. image_size - check the read_image function
    # 3. image_channels - check the read_image function
    # 4. num_classes - check the read_mask function
    # 5. reduce_mask_dims - check the read_mask function

    # 6. batch_size
    if not isinstance(batch_size, int):
        raise ValueError('The `batch_size` argument should be integer.')
    
    # 7. cache dataset
    if not isinstance(cache_dataset, bool):
        raise ValueError("The `cache_dataset` argument should either be "
                         "True or False.")
    # 8. shuffle dataset
    if not isinstance(shuffle_dataset, bool):
        raise ValueError("The `shuffle_dataset` argument should either be "
                         "True or False.")
    # 9. prefetch_dataset
    if not isinstance(prefetch_dataset, bool):
        raise ValueError("The `prefetch_dataset` argument should either be "
                         "True or False.")
    # 10. preprocess_function
    if not preprocess_function is None:
        if not isinstance(preprocess_function, object):
            raise ValueError("The `preprocess_function` argument should either "
                             "be a custom function or class.")

    #--------------------------------------------------------------------------#
    # Build data input pipeline function
    #--------------------------------------------------------------------------#
    # 1. Create a function to read images and masks into arrays.
    def read_image_and_mask(image_path, mask_path):
        """
        Reads and resizes both the image and its segmentation label (mask) from 
        their filepaths and return their equivalent tensor/array representation.  

        Args:
            image_path: path to the image.
            mask_path: path to the mask/segmentation label.

        Returns:
            image: image tensor representation of shape 
                   (image_height, image_width, image_channels)    
            mask: mask/segmentation label tensor representation of shape 
                   (image_height, image_width, image_channels)

        """

        image = read_image(image_path, 
                           image_size=image_size, 
                           image_channels=image_channels)

        mask = read_mask(mask_path, 
                         mask_size=image_size, 
                         num_classes=num_classes, 
                         reduce_mask_dims=reduce_mask_dims)

        return image, mask
    
    # 2. Load images and their masks using predefined arguments.
    buffer_size = len(image_paths)
    image_list = tf.constant(image_paths)
    mask_list = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    if shuffle_dataset:
        dataset = dataset.shuffle(buffer_size=buffer_size, 
                                  reshuffle_each_iteration=True)
    dataset = dataset.map(read_image_and_mask, 
                          num_parallel_calls=tf.data.AUTOTUNE)
    if preprocess_function:
        dataset = dataset.map(preprocess_function, 
                              num_parallel_calls=tf.data.AUTOTUNE)
    if cache_dataset:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size, 
                            num_parallel_calls=tf.data.AUTOTUNE)
    if prefetch_dataset:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset



################################################################################
# Data Augmentation Class
################################################################################
class Augment(tf.keras.layers.Layer):
    def __init__(self, augment_type="horizontal_and_vertical", seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(
        mode=augment_type, seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(
        mode=augment_type, seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


################################################################################
# Data Augmentation Function
################################################################################
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import save_img
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)
   
def augment_data(image_paths, mask_paths, save_path, image_size, augment=True):
    
    """
    Augments and save images and their masks/segmentation labels in a
    specific folder/directory.
    
    Args:
        image_paths: a list containing paths to the images
        mask_paths: a list containing paths to the images' segmenatations
        save_pathe: directory to save augmented images and masks.
        image_size: size at which the images and their segmenatations 
                    will be processed
        augment: (optional) boolean to augment dataset or not.
    Returns:
        Augmented images and their masks/segmentations
    """
    """ Performing data augmentation. """
    
    crop_size  = (image_size[0]-32, image_size[1]-32)
    
    for image, mask in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        name = image.split("/")[-1].split(".")
        image_name = name[0]
        image_extension = name[1]

        name = mask.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extension = name[1]       
        
        # Read images and masks from their filepaths using the read_image
        # and read_mask functions
        x = read_image(image, 
                       image_size=image_size, 
                       image_channels=3)

        y = read_mask(mask_path, 
                      mask_size=image_size, 
                      num_classes=3, 
                      reduce_mask_dims=True)
        
        try:
            h, w, c = x.shape
        except Exception as e:
            image = image[:-1]
            x, y = read_data(image, mask)
            h, w, c = x.shape     
    
    
        if augment == True:
            ## Center Crop
            aug = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            ## Crop
            x_min = 0
            y_min = 0
            x_max = x_min + image_size[0]
            y_max = y_min + image_size[1]

            aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            ## Random Rotate 90 degree
            aug = RandomRotate90(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            ## Transpose
            aug = Transpose(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            ## ElasticTransform
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            ## Grid Distortion
            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            ## Optical Distortion
            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']

            ## Vertical Flip
            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x8 = augmented['image']
            y8 = augmented['mask']

            ## Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x9 = augmented['image']
            y9 = augmented['mask']

            ## Grayscale
            x10 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y10 = y

            ## Grayscale Vertical Flip
            aug = VerticalFlip(p=1)
            augmented = aug(image=x10, mask=y10)
            x11 = augmented['image']
            y11 = augmented['mask']

            ## Grayscale Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x10, mask=y10)
            x12 = augmented['image']
            y12 = augmented['mask']

            ## Grayscale Center Crop
            aug = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])
            augmented = aug(image=x10, mask=y10)
            x13 = augmented['image']
            y13 = augmented['mask']

            ##
            aug = RandomBrightnessContrast(p=1)
            augmented = aug(image=x, mask=y)
            x14 = augmented['image']
            y14 = augmented['mask']

            aug = RandomGamma(p=1)
            augmented = aug(image=x, mask=y)
            x15 = augmented['image']
            y15 = augmented['mask']

            aug = HueSaturationValue(p=1)
            augmented = aug(image=x, mask=y)
            x16 = augmented['image']
            y16 = augmented['mask']

            aug = RGBShift(p=1)
            augmented = aug(image=x, mask=y)
            x17 = augmented['image']
            y17 = augmented['mask']

            aug = RandomBrightness(p=1)
            augmented = aug(image=x, mask=y)
            x18 = augmented['image']
            y18 = augmented['mask']

            aug = RandomContrast(p=1)
            augmented = aug(image=x, mask=y)
            x19 = augmented['image']
            y19 = augmented['mask']

            aug = MotionBlur(p=1, blur_limit=7)
            augmented = aug(image=x, mask=y)
            x20 = augmented['image']
            y20 = augmented['mask']

            aug = MedianBlur(p=1, blur_limit=10)
            augmented = aug(image=x, mask=y)
            x21 = augmented['image']
            y21 = augmented['mask']

            aug = GaussianBlur(p=1, blur_limit=10)
            augmented = aug(image=x, mask=y)
            x22 = augmented['image']
            y22 = augmented['mask']

            aug = GaussNoise(p=1)
            augmented = aug(image=x, mask=y)
            x23 = augmented['image']
            y23 = augmented['mask']

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x24 = augmented['image']
            y24 = augmented['mask']

            aug = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x25 = augmented['image']
            y25 = augmented['mask']

            images = [
                x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                x21, x22, x23, x24, x25
            ]
            masks  = [
                y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
                y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
                y21, y22, y23, y24, y25
            ]

        else:
            images = [x]
            masks  = [y]
            idx = 0
        
        for image, mask in zip(augmented_images, augmented_masks):
            tmp_image_name = f"{image_name}_{idx}.{image_extension}"
            tmp_mask_name = f"{mask_name}_{idx}.{mask_extension}" 

            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            save_img(image_path, image)
            save_img(mask_path, mask, scale=False)

            idx += 1
        