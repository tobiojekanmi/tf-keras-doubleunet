import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.python.lib.io import file_io
import warnings

from .lib.encoders import encoder_models
from .lib.utils import get_encoder_model
from .lib.utils import get_encoder_model_output_layer
from .lib.utils import get_skip_connection_layers



################################################################################
# Squeeze and Excite Block Function
################################################################################
def squeeze_excite_block(se_inputs, ratio=8):
    """
    Squeeze-and-Excitation Network Block
    Args:
        se_inputs: squeeze and excitation network input
        ratio: filter reduction ratio
    Returns:
        x: squeeze and excited output
    """
    init = se_inputs
    channel_axis = -1
    num_filters = se_inputs.shape[channel_axis]
    se_shape = (1, 1, num_filters)

    se = GlobalAveragePooling2D()(se_inputs)
    se = Reshape(se_shape)(se)
    se = Dense(num_filters//ratio, activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se_inputs)
    se = Dense(num_filters, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se_inputs)

    x = Multiply()([se_inputs, se])

    return x




################################################################################
# Default Convolution Block Function
################################################################################
def convolution_block(block_input,
                      num_filters=256,
                      kernel_size=3,
                      dilation_rate=1,
                      padding="same",
                      use_bias=False,
                      use_batchnorm=True,
                      activation='relu',
                      use_squeeze_and_excitation=True):
    """
    Instantiates default convolution block.

    Args:
        block_input: convolution block input
        num_filters: number of convolution filters to use
        kernel_size: kernel size to use
        dilation_rate: (optional) convolution dilation rate.
                       Default to 1.
        padding: (optional) type of padding to use.
                 Default to "same"
        use_bias: (optional) boolean specifying whether to use bias
                  or not. Default to False.
        use_batchnorm: (optional) boolean specifying whether to use
                       BatchNormalization layer or not. Default to True.
        activation: (optional) activation to use after each convolution
                    operation. Default to 'relu'.
        use_squeeze_and_excitation: (optional) boolean specifying whether
                                    to use squeeze and excitation block after
                                    convolution. Default to True.

    Returns:
        x: convolution block output

    """

    if activation == None:
        activation = 'linear'

    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               padding="same",
               use_bias=use_bias,
               kernel_initializer=tf.keras.initializers.HeNormal()
              )(block_input)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # Squeeze and excite block output
    if use_squeeze_and_excitation:
        x = squeeze_excite_block(x, ratio=8)

    return x



################################################################################
# DoubleU-Net Encoder Function
################################################################################
def encoder(encoder_type='Default',
            input_tensor=None,
            encoder_weights=None,
            encoder_freeze=False,
            num_blocks=5,
            encoder_filters=[32, 64, 128, 256, 512]):
    """
    Instantiates the encoder architecture for the DoubleU-Net segmentation model.

    Args:
        encoder_type: type of model to build upon. One of 'Default',
                      'DenseNet121', 'DenseNet169' 'EfficientNetB0',
                      'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                      'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
                      'EfficientNetB7', 'MobileNet', 'MobileNetV2',
                      'ResNet50', 'ResNet101', 'ResNet152',
                      'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                      'VGG16', 'VGG19'. Default encoder type is  'Default'.
        input_tensor: a tensorflow tensor input of a tuple containg image
                      width, height and channels respectively.
        encoder_weights: One of `None` (random initialization), `imagenet`,
                         or the path to the weights file to be loaded.
                         Default to None.
        encoder_freeze: Boolean Specifying whether to train encoder parameters
                        or not. Default to False.
        num_blocks: number of blocks to use for each encoder. Default to 5.
        encoder_filters: a list or tuple containing the number of filters to
                          use for each encoder block.
                          Default to [32, 64, 128, 256, 512].
    Returns:
        encoder_model: keras model
        encoder_model_output: Encoder model output.
        skip_connection_layers: Skip connection layers/encoder
                                block outputs before pooling.
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. encoder_type
    if encoder_type.lower() != 'default':
        if not encoder_type in encoder_models:
            raise ValueError(
                "The `encoder_type` argument is not not properly "
                "defined. Kindly use one of the following encoder "
                "names: 'Default', 'DenseNet121', 'DenseNet169', "
                "'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', "
                "'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', "
                "'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB67', "
                "'MobileNet', 'MobileNetV2', 'ResNet50', 'ResNet101', "
                "'ResNet152', 'ResNet50V2', 'ResNet101V2', "
                "'ResNet152V2', 'VGG16', VGG19'.")

    # 2. encoder_weights
    if encoder_type.lower() == 'default':
        if not (encoder_weights in {None} or
                file_io.file_exists_v2(encoder_weights)):
            warnings.warn('Ensure the `encoder_weights` argument is either '
                          '`None` (random initialization), '
                          'or the path to the weights file to be loaded. ')
    else:
        if not (encoder_weights in {'imagenet', None} or
                file_io.file_exists_v2(encoder_weights)):
            warnings.warn('The `encoder_weights` argument should be either '
                          '`None` (random initialization), `imagenet` '
                          '(pre-training on ImageNet), or the path to the '
                          'weights file to be loaded.')

    # 3. encoder_freeze
    if not isinstance(encoder_freeze, bool):
        raise ValueError("The `encoder_freeze` argument "
                         "should either be True or False.")

    # 4. num_blocks
    if not isinstance(num_blocks, int):
        raise ValueError('The `num_blocks` argument should be integer')
    elif num_blocks <= 0:
        raise ValueError('The `num_blocks` argument cannot be '
                         'less than or equal to zero')

    # 5. encoder_filters
    if not (isinstance(encoder_filters, tuple) or
            isinstance(encoder_filters, list)):
        raise ValueError('The `encoder_filters` argument should be a list of '
                         'tuple.')
    elif len(encoder_filters) <= 0:
        raise ValueError('The `encoder_filters` argument cannot be an empty '
                         'list.')
    elif len(encoder_filters) < num_blocks:
        raise ValueError('The items of the `encoder_filters` argument cannot '
                         'be less than the `num_blocks` argument.')

    #--------------------------------------------------------------------------#
    # Build the encoding blocks from the arguments specified
    #--------------------------------------------------------------------------#
    # 1. Default encoding blocks
    #--------------------------------------------------------------------------#
    if encoder_type.lower() == 'default':
        x = input_tensor
        kernel_size = 3
        pool_size = 2
        skip_connection_layers = []

        # Design the model
        for filter_id in range(num_blocks):
            num_filters = encoder_filters[filter_id]
            x = convolution_block(x,
                                  num_filters=num_filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=1,
                                  padding="same",
                                  use_bias=False,
                                  use_batchnorm=True,
                                  activation='relu')

            x = convolution_block(x,
                                  num_filters=num_filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=1,
                                  padding="same",
                                  use_bias=False,
                                  use_batchnorm=True,
                                  activation='relu')

            # save block output layer before pooling
            skip_connection_layers.append(x)

            x = MaxPooling2D((pool_size, pool_size))(x)

        encoder_model_output = x

        # Create model
        encoder_model = Model(input_tensor, encoder_model_output, name='model_1')

    #------------------------------------------------------------------------------#
    # 2. Pretrained model encoding blocks
    #------------------------------------------------------------------------------#
    else:
        encoder_model = get_encoder_model(encoder_type,
                                          input_tensor,
                                          encoder_weights)
        skip_connection_layers = get_skip_connection_layers(encoder_type,
                                                            encoder_model)
        encoder_model_output = get_encoder_model_output_layer(encoder_type,
                                                              encoder_model,
                                                              num_blocks)

    # Make the model parameters trainable or non trainable
    encoder_model.trainable = not(encoder_freeze)

    return encoder_model, encoder_model_output, skip_connection_layers




################################################################################
# Atrous/Dilated Spatial Pyramid Pooling Function
################################################################################
def DilatedSpatialPyramidPooling(dspp_input,
                                 num_filters=256,
                                 dilation_rates=[1,6,12,18]):
    """
    Instantiates the Atrous/Dilated Spatial Pyramid Pooling (ASPP/DSPP)
    architecture for the DoubleU-Net segmentation model.

    Args:
        dspp_input: DSPP input or encoder model ouput
        num_filters: Number of convolution filters.
        dilation_rates: a list containing dilate rates.
    Returns:
     dspp_output: dspp block output
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. num_filters
    if not isinstance(num_filters, int):
        num_filters = int(num_filters)
        warnings.warn("The `num_filters` argument is not an integer. "
                      "It will be rounded to the nearest integer "
                      "(if it's data type is float). ")

    # 2. dilation_rates
    if not (isinstance(dilation_rates, tuple) or
            isinstance(dilation_rates, list)):
        raise ValueError("The `dilation_rates` argument should either a "
                         "list or tuple.")

    #--------------------------------------------------------------------------#
    # Build the DSSP function from the arguments specified
    #--------------------------------------------------------------------------#
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x,
                          num_filters=num_filters,
                          kernel_size=1,
                          use_squeeze_and_excitation=False)
    out_pool = UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
                            interpolation="bilinear")(x)

    out_1 = convolution_block(dspp_input,
                              num_filters=num_filters,
                              kernel_size=1,
                              dilation_rate=dilation_rates[0],
                              use_squeeze_and_excitation=False)
    out_6 = convolution_block(dspp_input,
                              num_filters=num_filters,
                              kernel_size=3,
                              dilation_rate=dilation_rates[1],
                              use_squeeze_and_excitation=False)
    out_12 = convolution_block(dspp_input,
                               num_filters=num_filters,
                               kernel_size=3,
                               dilation_rate=dilation_rates[2],
                               use_squeeze_and_excitation=False)
    out_18 = convolution_block(dspp_input,
                               num_filters=num_filters,
                               kernel_size=3,
                               dilation_rate=dilation_rates[3],
                               use_squeeze_and_excitation=False)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])

    dspp_output = convolution_block(x,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    dilation_rate=1,
                                    use_squeeze_and_excitation=False)

    return dspp_output



################################################################################
# DoubleU-Net Decoder Function
################################################################################
def decoder(num_classes,
            decoder_input,
            skip_connection_layers_1,
            skip_connection_layers_2=None,
            decoder_type='upsampling',
            num_blocks=5,
            decoder_filters=[512, 256, 128, 64, 32],
            num_decoder_block_conv_layers=1,
            decoder_activation='relu',
            decoder_use_skip_connection=True,
            decoder_use_batchnorm=True,
            decoder_dropout_rate=0,
            output_activation=None):
    """
    Instantiates decoder architecture for the DoubleU-Net segmmantation model.
    Args:
        num_classes: number of the segmentation classes.
        decoder_input: decoder input or ASPP/DSPP block output.
        skip_connection_layers_1: a list or tuple containing the skip
                                connection output layers from the first
                                encoder model.
        skip_connection_layers_2: a list or tuple containing the skip
                                connection output layers from the second
                                encoder model. Only useful for Decoder 2.
        decoder_type: (optional) one of 'transpose' (to use Conv2DTanspose
                      operation for deconvolution operation) or 'upsampling'
                      (to use UpSampling2D operation for deconvolution
                      operation). Default to upsampling.
        num_blocks: (optional) number of encoder/decoder blocks to use.
                            Default to 5.
        decoder_filters: (optional) a list containing filter sizes for each
                          decoder block. Default to [512, 256, 128, 64, 32].
        num_decoder_block_conv_layers: (optional) number of additional
                                        convolution layersfor each decoder
                                        block. Default to 1.
        decoder_activation: (optional) decoder activation name or function.
                                       Default to 'relu'.
        decoder_use_skip_connection: (optional) one of True (to use skip
                                     connections) or False (not to use skip
                                     connections). Default to True.
        decoder_use_batchnorm: (optional) boolean to specify whether to
                                use BatchNormalization or not. Default to True.
        decoder_dropout_rate: (optional) dropout rate. Float between 0 and 1.
        output_activation: (optional) activation for output layer.
                            Default is either 'sigmoid' or 'softmax' based on
                            the value of the 'num_classes' argument.
    Returns:
        x: decoder output
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. number of classes specified
    if num_classes < 2:
        raise ValueError("The `num_classes` argument cannot be less than 2.")
    elif not (isinstance(num_classes, int)):
        raise ValueError("The `num_classes` argument can only be an integer.")

    # 2. decoder_type
    if not decoder_type.lower() in {'upsampling', 'transpose'}:
        raise ValueError("The `num_classes` argument can only be one of "
                         "'upsampling', or 'transpose'.")

    # 3. num_blocks
    if not isinstance(num_blocks, int):
        raise ValueError('The `num_blocks` argument should be integer')
    elif num_blocks <= 0:
        raise ValueError('The `num_blocks` argument cannot be '
                         'less than or equal to zero')

    # 4. decoder_filters
    if not (isinstance(decoder_filters, tuple) or
            isinstance(decoder_filters, list)):
        raise ValueError('The `decoder_filters` argument should be a list of '
                         'tuple.')
    elif len(decoder_filters) <= 0:
        raise ValueError('The `decoder_filters` argument cannot be an empty '
                         'list.')
    elif len(decoder_filters) < num_blocks:
        raise ValueError('The items of the `decoder_filters` argument cannot '
                         'be less than the `num_blocks` argument.')

    # 5. num_decoder_block_conv_layers
    if not isinstance(num_decoder_block_conv_layers, int):
        raise ValueError('The `num_decoder_block_conv_layers` '
                         'argument should be integer')
    elif num_decoder_block_conv_layers <= 0:
        raise ValueError('The `num_decoder_block_conv_layers` '
                         'argument cannot be less than or equal to zero')

    # 6. decoder_activation=None
    if decoder_activation == None:
        decoder_activation = 'relu'

    # 7. decoder_use_skip_connection
    if not isinstance(decoder_use_skip_connection, bool):
        raise ValueError("The `decoder_use_skip_connection` argument should "
                         "be either True or False.")

    # 8. decoder_use_batchnorm
    if not isinstance(decoder_use_batchnorm, bool):
        raise ValueError("The `decoder_use_batchnorm` argument should either "
                         "be True or False.")

    # 9. decoder_dropout_rate
    if not (isinstance(decoder_dropout_rate, int) or
            isinstance(decoder_dropout_rate, float)):
        raise ValueError('The `decoder_use_dropout` argument should be an '
                         'integer or float between 0 and 1')
    elif decoder_dropout_rate < 0 or decoder_dropout_rate > 1:
        raise ValueError("The `decoder_use_dropout` argument cannot be less "
                         "than 0 or greater than 1.")

    # 10. output activation
    if output_activation == None:
        if num_classes == 2:
            output_activation = 'sigmoid'
        else:
            output_activation = 'softmax'

    #--------------------------------------------------------------------------#
    # Build the decoder blocks from the arguments specified
    #--------------------------------------------------------------------------#
    ## Decoder Blocks
    decoder_filters = decoder_filters[-num_blocks:]
    skip_connection_layers_1 = skip_connection_layers_1[0:num_blocks]
    skip_connection_layers_1.reverse()
    if skip_connection_layers_2 != None:
        skip_connection_layers_2 = skip_connection_layers_2[0:num_blocks]
        skip_connection_layers_2.reverse()
    x = decoder_input
    if decoder_type.lower() == 'transpose':
        for decoder_block_id in range(num_blocks):
            r1 = skip_connection_layers_1[decoder_block_id]
            if skip_connection_layers_2 != None:
                r2 = skip_connection_layers_2[decoder_block_id]
            num_filters = decoder_filters[decoder_block_id]
            x = Conv2DTranspose(num_filters, 3, strides=(2, 2), padding="same")(x)
            if decoder_use_skip_connection:
                x = concatenate([x, r1], axis=-1)
                if skip_connection_layers_2 != None:
                    x = concatenate([x, r2], axis=-1)
            for block_layer_id in range(num_decoder_block_conv_layers):
                x = convolution_block(x,
                                      num_filters=num_filters,
                                      kernel_size=(3, 3),
                                      dilation_rate=1,
                                      padding="same",
                                      use_bias=False,
                                      use_batchnorm=decoder_use_batchnorm,
                                      activation=decoder_activation)

    elif decoder_type.lower() == 'upsampling':
        for decoder_block_id in range(num_blocks):
            r1 = skip_connection_layers_1[decoder_block_id]
            if skip_connection_layers_2 != None:
                r2 = skip_connection_layers_2[decoder_block_id]
            num_filters = decoder_filters[decoder_block_id]
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(num_filters, (3, 3), padding='same')(x)
            x = Activation(decoder_activation)(x)
            if decoder_use_skip_connection:
                x = concatenate([x, r1], axis=-1)
                if skip_connection_layers_2 != None:
                    x = concatenate([x, r2], axis=-1)
            for block_layer_id in range(num_decoder_block_conv_layers):
                x = convolution_block(x,
                                      num_filters=num_filters,
                                      kernel_size=(3, 3),
                                      dilation_rate=1,
                                      padding="same",
                                      use_bias=False,
                                      use_batchnorm=decoder_use_batchnorm,
                                      activation=decoder_activation)

    x = Conv2D(decoder_filters[-1], 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)

    x = Dropout(decoder_dropout_rate)(x)

    x = Conv2D(filters=num_classes, kernel_size=(1, 1),
               activation=output_activation,
               padding='same')(x)

    return x



#------------------------------------------------------------------------------#
# DoubleU-Net Model
#------------------------------------------------------------------------------#
def doubleunet(num_classes,
               input_shape=(224, 224, 3),
               model_weights=None,
               num_blocks=5,
               encoder_one_type='Default',
               encoder_one_weights=None,
               encoder_one_freeze=False,
               encoder_one_filters=[32, 64, 128, 256, 512],
               dspp_one_filters=256,
               decoder_one_type='upsampling',
               num_decoder_one_block_conv_layers=1,
               decoder_one_filters=[512, 256, 128, 64, 32],
               decoder_one_activation=None,
               decoder_one_use_skip_connection=True,
               decoder_one_use_batchnorm=True,
               decoder_one_dropout_rate=0,
               output_one_activation=None,
               encoder_two_type='Default',
               encoder_two_weights=None,
               encoder_two_freeze=False,
               encoder_two_filters=[32, 64, 128, 256, 512],
               dspp_two_filters=256,
               decoder_two_type='upsampling',
               decoder_two_filters=[512, 256, 128, 64, 32],
               num_decoder_two_block_conv_layers=1,
               decoder_two_activation=None,
               decoder_two_use_skip_connection=True,
               decoder_two_use_batchnorm=True,
               decoder_two_dropout_rate=0,
               output_two_activation=None):

    """
    Merge the doubleunet_encoder and doubleunet_decoder functions to instantiate
    the doubleunet architecture for semantic segmantation tasks.

    Args:
            num_classes: number of the segmentation classes.
            input_shape: a tuple containing image height, width and channels
                         respectively. Default to (224,224,3).
            model_weights: (optional) link to pre-trained weights.
            num_blocks: (optional) number of encoder and decoder blocks.
                        Default to 5.

            ############################ Encoder Blocks ########################
            encoder_one_type & encoder_two_type:
                        type of model to build upon. One of 'Default',
                        'DenseNet121', 'DenseNet169' 'EfficientNetB0',
                        'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                        'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
                        'EfficientNetB7', 'MobileNet', 'MobileNetV2',
                        'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                        'VGG16', 'VGG19'. Default encoder type is  'Default'.
            encoder_one_weights & encoder_two_weights:
                        (optional) pre-trained weights for encoder function.
                        One of None (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded
            encoder_one_freeze & encoder_two_freeze:
                        (optional) boolean to specify whether to train
                        encoder model parameters or not. Default is False.
            encoder_one_filters & encoder_two_filters:
                        (optional) a list containing number of filters to use
                        for each encoder convolution blocks.
                        Default to [32, 64, 128, 256, 512].

            ############################ DSPP Blocks ###########################
            dspp_one_filters & dspp_two_filters:
                        (optional) a list containing number of filters to use
                        for each DSSP block. Default to 256.

            ############################# Decoder Blocks #######################
            decoder_one_type & decoder_two_type:
                        (optional) one of 'transpose' (to use Conv2DTanspose
                        operation for upsampling operation) or 'upsampling' (to
                        use UpSampling2D operation for upsampling operation).
                        Default to upsampling.
            decoder_one_filters & decoder_two_filters:
                        (optional) a list containing number of filters to use
                        for each decoder convolution blocks.
                        Default to [512, 256, 128, 64, 32].
            num_decoder_one_blocks & num_decoder_two_blocks:
                        (optional) number of decoder blocks to use. Default to 5.
            decoder_one_filters & decoder_two_filters:
                        (optional) a list containing filter sizes for each
                        decoder block. Default to [32, 64, 128, 256, 512].
            num_decoder_one_block_conv_layers & num_decoder_two_block_conv_layers:
                        (optional) number of convolution layers for each decoder
                        block (i.e. number of Conv2D layers after upsampling
                        layers). Default is 1.
            decoder_one_activation & decoder_two_activation:
                        (optional) decoder activation name or function.
            decoder_one_use_skip_connection & decoder_two_use_skip_connection:
                        (optional) one of True (to use residual/skip connections)
                        or False (not to use residual/skip connections).
                        Default to True.
            decoder_use_batchnorm:
                        (optional) boolean to specify whether decoder layers
                        should use BatchNormalization or not.
                        Default to False.
            decoder_dropout_rate:
                        (optional) dropout rate. Float between 0 and 1.
            output_activation:
                        (optional) activation for output layer.
                        Default is either 'sigmoid' or 'softmax' based on
                        the value of the 'num_classes' argument.
    Returns:
        model: keras double-unet segmentation model
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. num_classes - check doubleunet_decoder functon
    # 2. encoder_type - check doubleunet_encoder functon
    # 3. input_shape - check doubleunet_encoder functon
    # 2. input_shape
    if not isinstance(input_shape, tuple):
        raise ValueError("The `input_shape` argument should a tuple containing "
                         "the image width, height and channels respectively.")
    if not len(input_shape) == 3:
        warnings.warn("The `input_shape` argument should be a tuple containing "
                      "three integer values for each of the image width, "
                      "height, and channels respectively.")

    # 4. model_weights
    if not (model_weights in {None} or file_io.file_exists_v2(model_weights)):
        warnings.warn('The `model_weights` argument should either be '
                      '`None` (random initialization), '
                      'or the path to the weights file to be loaded.')

    # 5. encoder_weights - check doubleunet_encoder functon
    # 6. encoder_freeze - check doubleunet_encoder functon
    # 7. num_bottleneck_conv_layers - check doubleunet_bottleneck functon
    # 8. num_bottleneck_conv_filters - check doubleunet_bottleneck functon
    # 9. bottleneck_use_batchnorm - check doubleunet_bottleneck functon
    # 10. num_decoder_blocks - check doubleunet_decoder functon
    # 11. decoder_type - check doubleunet_decoder functon
    # 12. decoder_filters - check doubleunet_decoder functon
    # 13. num_decoder_block_conv_layers - check doubleunet_decoder functon
    # 14. decoder_activation - check doubleunet_decoder functon
    # 15. decoder_use_skip_connection - check doubleunet_decoder functon
    # 16. decoder_use_batchnorm - check doubleunet_decoder functon
    # 17. decoder_dropout_rate - check doubleunet_decoder functon
    # 18. output_activation - check doubleunet_decoder functon

    #--------------------------------------------------------------------------#
    # Build Model
    #--------------------------------------------------------------------------#
    # Network 1
    #--------------------------------------------------------------------------#
    # 1. Get the encoder model, model output layer and skip connection layers
    input_1 = Input(shape=(input_shape), name='input_1')
    encoder_model_1, encoder_model_output_1, skip_connection_layers_1 = encoder(
        encoder_type=encoder_one_type,
        input_tensor=input_1,
        encoder_weights=encoder_one_weights,
        encoder_freeze=encoder_one_freeze,
        num_blocks=num_blocks,
        encoder_filters=encoder_one_filters
    )
    # 2. Get the ASPP/DSPP block output layer
    dspp_output_1 = DilatedSpatialPyramidPooling(
        dspp_input=encoder_model_output_1,
        num_filters=dspp_one_filters
    )

    # 3. Decoder blocks
    # Extend the model by adding the decoder blocks
    output_1 = decoder(
        num_classes=num_classes,
        decoder_input=dspp_output_1,
        skip_connection_layers_1 = skip_connection_layers_1,
        skip_connection_layers_2= None,
        decoder_type=decoder_one_type,
        num_blocks=num_blocks,
        decoder_filters=decoder_one_filters,
        num_decoder_block_conv_layers=num_decoder_one_block_conv_layers,
        decoder_activation=decoder_one_activation,
        decoder_use_skip_connection=decoder_one_use_skip_connection,
        decoder_use_batchnorm=decoder_one_use_batchnorm,
        decoder_dropout_rate=decoder_one_dropout_rate,
        output_activation=output_one_activation)

    # Rename encoder model one layer names to avoid none of the layers from
    # encoders one and two are the same.
    enc_1_layers = [layer for layer in
                    Model(encoder_model_1.inputs, output_1).layers]
    for layer in enc_1_layers:
        layer._name = layer._name + str("_a")

    #--------------------------------------------------------------------------#
    # Network 2
    #--------------------------------------------------------------------------#
    input_2 = Concatenate(axis=-1, name='input_2')([output_1, input_1])
    # 1. Get the encoder model, model output layer and skip connection layers
    encoder_model_2, encoder_model_output_2, skip_connection_layers_2 = encoder(
        encoder_type=encoder_two_type,
        input_tensor=input_2,
        encoder_weights=encoder_two_weights,
        encoder_freeze=encoder_two_freeze,
        num_blocks=num_blocks,
        encoder_filters=encoder_two_filters
    )

    # 2. Get the ASPP/DSPP block output layer
    dspp_output_2 = DilatedSpatialPyramidPooling(
        dspp_input=encoder_model_output_2,
        num_filters=dspp_two_filters
    )

    # 3. Decoder blocks
    # Extend the model by adding the decoder blocks
    output_2 = decoder(
        num_classes=num_classes,
        decoder_input=dspp_output_2,
        skip_connection_layers_1 = skip_connection_layers_1,
        skip_connection_layers_2 = skip_connection_layers_2,
        decoder_type=decoder_two_type,
        num_blocks=num_blocks,
        decoder_filters=decoder_two_filters,
        num_decoder_block_conv_layers=num_decoder_two_block_conv_layers,
        decoder_activation=decoder_two_activation,
        decoder_use_skip_connection=decoder_two_use_skip_connection,
        decoder_use_batchnorm=decoder_two_use_batchnorm,
        decoder_dropout_rate=decoder_two_dropout_rate,
        output_activation=output_two_activation
    )

    # Rename encoder model two layer names if both encoder one and two are the same
    enc_1_layers = [layer for layer in
                    Model(encoder_model_1.inputs, output_1).layers]
    enc_2_layers = [layer for layer in
                    Model(encoder_model_2.inputs, output_2).layers
                    if layer not in enc_1_layers]
    for layer in enc_2_layers:
        layer._name = layer._name + str("_b")

    outputs = Add()([output_1, output_2])
    inputs = encoder_model_1.inputs

    ## Image Segmentation Model
    model = Model(inputs, outputs)

    return model
