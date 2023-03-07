
import tensorflow as tf
import os, shutil
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'



CLASS_NAMES = ['Bad','Good']
MODEL_IMG_SIZE = [224, 224]  # What mobilenet expects

def MobileNetV2(IMG_CHANNELS, trainable=False):
    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = tuple(MODEL_IMG_SIZE) + (IMG_CHANNELS,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    
    base_model.trainable = trainable
    # must be othervise function return compresed version of model
    # base_model = [layer for layer in base_model.layers] 
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input    

    # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
    # Prediction layer
    prediction_layer = tf.keras.layers.Dense(1)
    
    inputs = tf.keras.Input(shape=(IMG_SHAPE))
    x = preprocess_input(inputs)
    x = base_model(x, training=trainable)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    # model
    return model


