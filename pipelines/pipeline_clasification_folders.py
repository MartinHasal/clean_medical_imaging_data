# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:56:41 2023

@author: Martin Hasal

Second version of programe, after oneFileSolution with 90% acc on test

main pipeline for medical image cleaning by classification neural network
Main goal is to distinguis good and bad medical images by 
MobileNetV2 CNN with fine tunning

datasets are loaded by ImageDataGenerator, it future as the input dataset will be
stable (now in progress), it be transfered to tf recods by external function 
"""




import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os, shutil
import argparse
#import hypertune
from distutils.util import strtobool
from tensorflow.data.experimental import AUTOTUNE     

from utils.util import cleanup_dir, create_strategy
#from ingest.tfrecords import *
from models.model_mobilenetv2 import MobileNetV2
from utils.plots import *
from models.model_mobilenetv2 import MODEL_IMG_SIZE
from keras_preprocessing.image import ImageDataGenerator



def presision(predictions, probabilities):
    """ 
    Computes precision of prediction for binary clasissifier
    """
    pre = predictions.numpy()
    pro = probabilities.numpy()
    return 1 - np.abs(pre - pro)

def train_and_evaluate(opts, strategy=None):
    # calculate the image dimensions given that we have to center crop
    # to achieve the model image size
    IMG_HEIGHT, IMG_WIDTH = MODEL_IMG_SIZE
    print('Input images are resized to {}x{}'.format( IMG_HEIGHT, IMG_WIDTH ))
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    
    
    datagen = ImageDataGenerator(horizontal_flip=True)
    # load and iterate training dataset
    # opts['input_topdir'], 'valid' + opts['pattern']
    train_dataset = datagen.flow_from_directory("data/processed/Processed1/train", 
                                           class_mode='binary', 
                                           color_mode="rgb",
                                           target_size=MODEL_IMG_SIZE,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
    # load and iterate validation dataset
    validation_dataset = datagen.flow_from_directory("data/processed/Processed1/val", 
                                           class_mode='binary', 
                                           color_mode="rgb",
                                           target_size=MODEL_IMG_SIZE,
                                           # batch_size=opts['batch_size']
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
    # load and iterate test dataset
    test_datagen = ImageDataGenerator()
    test_dataset = test_datagen.flow_from_directory("data/processed/Processed1/test",  
                                           color_mode="rgb",
                                           target_size=MODEL_IMG_SIZE,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
    
    
    
    
    
    
    """
    print("Number of layers in the base model: ", len(model.layers))
    
    My motivation was to use fine-tunnig on last 30 layers after training the
    last layer, but ...
    If I create create model in outside function because 
    and I use a transfer learning to MobileNetV2 model and put it together
    wit all other layers it returns whole MobileNetV2 as single layer, 
    with 7 layers in total. I spend several hours how to fix it, but it
    gives an errror when I add layers one by one from original MobileNetV2.
    I tested create a model by Sequentional api, but the same mistake with 
    * Exception encountered when calling layer "block_2_add" (type Add).*
    There are two solution
    1, create model here with sepate inner base_function = MobileNetV2
    2, Use model with only seven layer are make it trainable at all layer
    
    I chosed 2, by internal testing it produces a better result then fine-tunnig
    (20 epochs last layer, 30 epochs last 30 layers),
    and train MobileNetV2 is not an issue. Alernativelly, I can add warm-up lr.    
    """
    
    
    
    if strategy:
        with strategy.scope():
            model = MobileNetV2(IMG_CHANNELS,trainable=True)
    else:
        model = MobileNetV2(IMG_CHANNELS,trainable=True)
    print(model.summary())
    

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=opts['lrate']),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    

    loss0, accuracy0 = model.evaluate(validation_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                        epochs=opts['num_epochs'],
                        validation_data=validation_dataset,
                        callbacks=[early_stopping_cb])
        
    # model training
    
   
    training_plot(['loss', 'accuracy'], history, 
                  os.path.join(opts['outdir'], 'training_plot.png'))
    
    ### test images
    image_batch, label_batch = test_dataset.next()
    probabilities = model.predict_on_batch(image_batch).flatten()
    # Apply a sigmoid since our model returns logits
    probabilities = tf.nn.sigmoid(probabilities)
    predictions = tf.where(probabilities < 0.5, 0, 1)
    precision = presision(predictions,probabilities)
    display_batch_of_images((image_batch/255, label_batch.astype(int)), predictions, precision)


    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    
       
    # report hyperparam metric
    # hpt = hypertune.HyperTune()
    # accuracy = np.max(history.history['val_accuracy']) # highest encountered
    # nepochs = len(history.history['val_accuracy'])
    # hpt.report_hyperparameter_tuning_metric(
    #     hyperparameter_metric_tag='accuracy',
    #     metric_value=accuracy,
    #     global_step=nepochs)
    # print("Reported hparam metric name=accuracy value={}".format(accuracy))
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Training parameters
    parser.add_argument(
        '--job-dir', help='Top-level output directory', required=True)
    parser.add_argument(
        '--pattern', help='Files in {input_topdir}/train to read',
        default='-0000[01]-*')
    parser.add_argument(
        '--num_epochs', help='How many times to iterate over training patterns',
        default=3, type=int)
    parser.add_argument(
        '--distribute', default='gpus_one_machine',
        help="""
            Has to be one of:
            * cpu
            * gpus_one_machine
            * gpus_multiple_machines
            * tpu_colab
            * tpu_caip
            * the actual name of the cloud_tpu
        """, type=str )
    parser.add_argument('--resume', dest='resume', action='store_true',
                       help="Starts from checkpoints in output directory")
    
    ## model parameters
    parser.add_argument(
        '--batch_size', help='Number of records in a batch', default=32, type=int)
    parser.add_argument(
        '--l1', help='L1 regularization', default=0., type=float)
    parser.add_argument(
        '--l2', help='L2 regularization', default=0., type=float)
    parser.add_argument(
        '--lrate', help='Adam learning rate', default=0.001, type=float)
    parser.add_argument(
        '--num_hidden', help='Number of nodes in last but one layer', default=16, type=int)
    parser.add_argument('--with_color_distort',
                        type=lambda x: bool(strtobool(x)), nargs='?', const=True, default=True,
                        help="Specify True or False. Default is True.")

    # parse arguments, and set up outdir based on job-dir
    # job-dir is set by CAIP, but outdir is what our code wants.
    args = parser.parse_args()
    opts = args.__dict__
    opts['outdir'] = opts['job_dir']
    print("Job Parameters={}".format(opts))
    
    # able to resume
    if not opts['resume']:
        cleanup_dir(os.path.join(opts['outdir'], 'chkpts'))
    
    # Train, evaluate, export
    strategy = create_strategy(opts['distribute'])  # gpus_one_machine=None due to autoshard warning
    train_and_evaluate(opts,strategy)
