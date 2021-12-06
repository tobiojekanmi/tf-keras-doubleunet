import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19


encoder_models = {
    # DenseNet
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,

    # EfficientNet
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'EfficientNetB2': EfficientNetB2,
    'EfficientNetB3': EfficientNetB3,
    'EfficientNetB4': EfficientNetB4,
    'EfficientNetB5': EfficientNetB5,
    'EfficientNetB6': EfficientNetB6,
    'EfficientNetB7': EfficientNetB7,

    # MobileNet
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,

    # ResNet
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'ResNet50V2': ResNet50V2,
    'ResNet101V2': ResNet101V2,
    'ResNet152V2': ResNet152V2,

    # VGG
    'VGG16': VGG16,
    'VGG19': VGG19
    }



model_output_layers = {
    # DenseNet
    'DenseNet121': ['conv1/relu',
                    'pool2_conv',
                    'pool3_conv',
                    'pool4_conv',
                    'relu'],

    'DenseNet169': ['conv1/relu',
                    'pool2_conv',
                    'pool3_conv',
                    'pool4_conv',
                    'relu'],

    # EfficientNet
    'EfficientNetB0': ['block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation',
                       'top_activation'],

    'EfficientNetB1': ['block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation',
                       'top_activation'],

    'EfficientNetB2': ['block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation',
                       'top_activation'],

    'EfficientNetB3': ['block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation',
                       'top_activation'],

    'EfficientNetB4': ['block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation',
                       'top_activation'],

    'EfficientNetB5': ['block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation',
                       'top_activation'],

    'EfficientNetB6': ['block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation',
                       'top_activation'],

    'EfficientNetB7': ['block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation',
                       'top_activation'],

    # MobileNet
    'MobileNet': ['conv_pw_1_relu',
                  'conv_pw_3_relu',
                  'conv_pw_5_relu',
                  'conv_pw_11_relu',
                  'conv_pw_13_relu'],

    'MobileNetV2': ['block_1_expand_relu',
                    'block_3_expand_relu',
                    'block_6_expand_relu',
                    'block_13_expand_relu',
                    'out_relu'],

    # ResNet
    'ResNet50': ['conv1_relu',
                 'conv2_block3_out',
                 'conv3_block4_out',
                 'conv4_block6_out',
                 'conv5_block3_out'],

    'ResNet101': ['conv1_relu',
                  'conv2_block3_out',
                  'conv3_block4_out',
                  'conv4_block23_out',
                  'conv5_block3_out'],

    'ResNet152': ['conv1_relu',
                  'conv2_block3_out',
                  'conv3_block8_out',
                  'conv4_block36_out',
                  'conv5_block3_out'],

    'ResNet50V2': ['conv1_conv',
                   'conv2_block3_1_relu',
                   'conv3_block4_1_relu',
                   'conv4_block6_1_relu',
                   'post_relu'],

    'ResNet101V2': ['conv1_conv',
                    'conv2_block3_1_relu',
                    'conv3_block4_1_relu',
                    'conv4_block23_1_relu',
                    'post_relu'],

    'ResNet152V2': ['conv1_conv',
                    'conv2_block3_1_relu',
                    'conv3_block8_1_relu',
                    'conv4_block36_1_relu',
                    'post_relu'],

    # VGG
    'VGG16': ['block1_pool',
              'block2_pool',
              'block3_pool',
              'block4_pool',
              'block5_pool'],
    'VGG19': ['block1_pool',
              'block2_pool',
              'block3_pool',
              'block4_pool',
              'block5_pool']
    }


skip_connection_layers = {
    # DenseNet
    'DenseNet121': ['',
                    'conv1/relu',
                    'pool2_conv',
                    'pool3_conv',
                    'pool4_conv'],
    'DenseNet169': ['',
                    'conv1/relu',
                    'pool2_conv',
                    'pool3_conv',
                    'pool4_conv'],

    # EfficientNet
    'EfficientNetB0': ['stem_conv_pad',
                       'block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation'],
    'EfficientNetB1': ['stem_conv_pad',
                       'block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation'],
    'EfficientNetB2': ['stem_conv_pad',
                       'block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation'],
    'EfficientNetB3': ['stem_conv_pad',
                       'block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation'],
    'EfficientNetB4': ['stem_conv_pad',
                       'block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation'],
    'EfficientNetB5': ['stem_conv_pad',
                       'block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation'],
    'EfficientNetB6': ['stem_conv_pad',
                       'block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation'],
    'EfficientNetB7': ['stem_conv_pad',
                       'block2a_expand_activation',
                       'block3a_expand_activation',
                       'block4a_expand_activation',
                       'block6a_expand_activation'],

    # MobileNet
    'MobileNet': ['conv1',
                  'conv_pw_1_relu',
                  'conv_pw_3_relu',
                  'conv_pw_5_relu',
                  'conv_pw_11_relu'],

    'MobileNetV2': ['Conv1',
                    'block_1_expand_relu',
                    'block_3_expand_relu',
                    'block_6_expand_relu',
                    'block_13_expand_relu'],

    # ResNet
    'ResNet50': ['conv1_pad',
                 'conv1_relu',
                 'conv2_block3_out',
                 'conv3_block4_out',
                 'conv4_block6_out'],

    'ResNet101': ['conv1_pad',
                  'conv1_relu',
                  'conv2_block3_out',
                  'conv3_block4_out',
                  'conv4_block23_out'],

    'ResNet152': ['conv1_pad',
                  'conv1_relu',
                  'conv2_block3_out',
                  'conv3_block8_out',
                  'conv4_block36_out'],

    'ResNet50V2': ['conv1_pad',
                   'conv1_conv',
                   'conv2_block3_1_relu',
                   'conv3_block4_1_relu',
                   'conv4_block6_1_relu'],

    'ResNet101V2': ['conv1_pad',
                    'conv1_conv',
                    'conv2_block3_1_relu',
                    'conv3_block4_1_relu',
                    'conv4_block23_1_relu'],

    'ResNet152V2': ['conv1_pad',
                    'conv1_conv',
                    'conv2_block3_1_relu',
                    'conv3_block8_1_relu',
                    'conv4_block36_1_relu'],

    # VGG
    'VGG16': ['block1_conv2',
              'block2_conv2',
              'block3_conv3',
              'block4_conv3',
              'block5_conv3'],

    'VGG19': ['block1_conv2',
              'block2_conv2',
              'block3_conv4',
              'block4_conv4',
              'block5_conv3']
    }
