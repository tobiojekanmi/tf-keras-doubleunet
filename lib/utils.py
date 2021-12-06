from .encoders import encoder_models
from .encoders import model_output_layers
from .encoders import skip_connection_layers



def get_encoder_model(encoder_type, input_tensor, encoder_weights):

    """
    Instantiates encoder model architecture function for the encoder_type
    specified.
    Args:
        encoder_type:   type of encoder function to use for model.
                        Kindly use 'Default' for default encoder or use
                        one of the following pre-trained models as encoders:
                        'DenseNet121', 'DenseNet169', 'DenseNet201',
                        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                        'EfficientNetB6', 'EfficientNetB67', 'MobileNet',
                        'MobileNetV2', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'VGG16',
                        'VGG19'
        input_tensor: Input tensor
        encoder_weights: encoder model weights
    Returns:
        encoder: encoder model function
    """
    #--------------------------------------------------------------------------#
    # Validate Arguments
    #--------------------------------------------------------------------------#
    if str(encoder_type) in encoder_models:
        encoder = encoder_models[encoder_type]
    else:
        raise ValueError("The `encoder_type` argument is not not properly defined."
                    " Kindly use one of the following encoder names: 'Default', "
                    "'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0',"
                    " 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', "
                    "'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', "
                    "'EfficientNetB67', 'MobileNet', 'MobileNetV2', 'ResNet50', "
                    "'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', "
                    "'ResNet152V2', 'VGG16', VGG19'")
    #--------------------------------------------------------------------------#
    # Build function
    #--------------------------------------------------------------------------#
    encoder_model = encoder(include_top=False,
                            weights=encoder_weights,
                            input_tensor=input_tensor)

    return encoder_model



def get_encoder_model_output_layer(encoder_type, encoder_model, num_blocks):
    """
    Gets the encoder model output layer.
    Args:
        encoder_type:   type of encoder function to use for model.
                        Kindly use 'Default' for default encoder or use
                        one of the following pre-trained models as encoders:
                        'DenseNet121', 'DenseNet169', 'DenseNet201',
                        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                        'EfficientNetB6', 'EfficientNetB67', 'MobileNet',
                        'MobileNetV2', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'VGG16',
                        'VGG19'
        encoder_model: keras model of the encoder_type argument
        num_blocks: number of encoder/decoder model blocks (as applicable)
    Returns:
        output_layer: encoder model output
    """
    #--------------------------------------------------------------------------#
    # Validate Arguments
    #--------------------------------------------------------------------------#
    if str(encoder_type) in encoder_models:
        encoder = encoder_models[encoder_type]
    else:
        raise ValueError("The `encoder_type` argument is not not properly defined. "
                    "Kindly use one of the following encoder names: 'Default', "
                    "'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', "
                    "'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', "
                    "'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', "
                    "'EfficientNetB67', 'MobileNet', 'MobileNetV2', 'ResNet50', "
                    "'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', "
                    "'ResNet152V2', 'VGG16', VGG19'.")
    #--------------------------------------------------------------------------#
    # Build Function
    #--------------------------------------------------------------------------#
    layer = model_output_layers[encoder_type][num_blocks-1]
    output_layer = encoder_model.get_layer(layer).output

    return output_layer



def get_skip_connection_layers(encoder_type, encoder_model):
    """
    Gets the encoder model skip connection layers.
    Args:
        encoder_type:   type of encoder function to use for model.
                        Kindly use 'Default' for default encoder or use
                        one of the following pre-trained models as encoders:
                        'DenseNet121', 'DenseNet169', 'DenseNet201',
                        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                        'EfficientNetB6', 'EfficientNetB67', 'MobileNet',
                        'MobileNetV2', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'VGG16',
                        'VGG19'
        encoder_model: keras model of the encoder_type argument
    Returns:
        model_skip_connection_layers: encoder model skip connection layers
    """
    #--------------------------------------------------------------------------#
    # Validate Arguments
    #--------------------------------------------------------------------------#
    if str(encoder_type) in encoder_models:
        encoder = encoder_models[encoder_type]
    else:
        raise ValueError("The `encoder_type` argument is not not properly defined. "
                    "Kindly use one of the following encoder names: 'Default', "
                    "'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', "
                    "'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', "
                    "'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', "
                    "'EfficientNetB67', 'MobileNet', 'MobileNetV2', 'ResNet50', "
                    "'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', "
                    "'ResNet152V2', 'VGG16', VGG19'.")
    #--------------------------------------------------------------------------#
    # Build Function
    #--------------------------------------------------------------------------#
    model_skip_connection_layer_names = skip_connection_layers[encoder_type]
    if model_skip_connection_layer_names[0] == '':
        first_layer_output = encoder_model.input
    else:
        first_layer_output = encoder_model.get_layer(
            model_skip_connection_layer_names[0]
        ).input


    model_skip_connection_layers = [encoder_model.get_layer(layer_name).output
                                    for layer_name in
                                    model_skip_connection_layer_names[1:]]
    model_skip_connection_layers.insert(0, first_layer_output)

    return model_skip_connection_layers
