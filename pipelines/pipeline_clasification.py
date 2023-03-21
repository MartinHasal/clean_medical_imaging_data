"""
Created on Fri Mar  3 14:56:41 2023

@author: Martin Hasal

Second version of programe, after oneFileSolution with 90% acc on test

main pipeline for medical image cleaning by classification neural network
Main goal is to distinguis good and bad medical images by
MobileNetV2 CNN with fine tunning

datasets are loaded by ImageDataGenerator, it future as the input dataset will
be stable (now in progress), it be transfered to tf recods by external function
"""
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


import numpy as np
import tensorflow as tf
import os, shutil
import argparse
#import hypertune

  

from utils.util import cleanup_dir, create_strategy
#from ingest.tfrecords import *
from utils.plots import *
from models.model_mobilenetv2 import MODEL_IMG_SIZE, MobileNetV2
from keras_preprocessing.image import ImageDataGenerator
from math import ceil


def presision(predictions, probabilities):
    """
    Computes precision of prediction for binary clasissifier
    """
    pre = predictions.numpy()
    pro = probabilities.numpy()
    return 1 - np.abs(pre - pro)


def train_and_evaluate(opts, strategy=None):
    """
    Function which handles inputs and train the model
    """
    IMG_CHANNELS = 3

    # load and iterate training dataset
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 rotation_range=70,
                                 width_shift_range=0.08,
                                 height_shift_range=0.08,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 vertical_flip=True,
                                 fill_mode='nearest')
    # load and iterate training dataset
    # opts['input_topdir'], 'valid' + opts['pattern']
    train_dataset = datagen.flow_from_directory("./data/processed/Classification/train",
                                                class_mode='binary',
                                                color_mode="rgb",
                                                target_size=MODEL_IMG_SIZE,
                                                batch_size=opts['batch_size'],
                                                shuffle=True
                                                )

    # load and iterate validation dataset
    validation_dataset = datagen.flow_from_directory("data/processed/Classification/val", 
                                                     class_mode='binary',
                                                     color_mode="rgb",
                                                     target_size=MODEL_IMG_SIZE,
                                                     batch_size=opts['batch_size'],
                                                     shuffle=True
                                                     )

    # load and iterate test dataset
    test_datagen = ImageDataGenerator()
    test_dataset = test_datagen.flow_from_directory("data/processed/Classification/test",
                                                    color_mode="rgb",
                                                    class_mode='binary', 
                                                    target_size=MODEL_IMG_SIZE,
                                                    batch_size=opts['batch_size'],
                                                    shuffle=True
                                                    )
    class_names_test = test_dataset.class_indices
    CLASSES = list(class_names_test.keys())

    # model load
    if strategy:
        with strategy.scope():
            model = MobileNetV2(IMG_CHANNELS, trainable=opts['trainable'])
    else:
        model = MobileNetV2(IMG_CHANNELS, trainable=opts['trainable'])

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=opts['lrate']),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy()]
                  )

    loss0, accuracy0 = model.evaluate(validation_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    # checkpoint and early stopping callbacks
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(opts['outdir'], 'chkpts'),
        monitor='val_binary_accuracy', mode='max',
        save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', mode='max',
        patience=10)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                     factor=0.2,
                                                     patience=4,
                                                     min_lr=0.00001,
                                                     mode='auto',
                                                     cooldown=4,
                                                     verbose=1)

    num_steps_per_epoch = None
    if (train_dataset.samples > 0):
        num_steps_per_epoch = train_dataset.samples // opts['batch_size']
        print("Model train for {} steps".format(num_steps_per_epoch))

    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=opts['num_epochs'],
                        steps_per_epoch=num_steps_per_epoch,
                        callbacks=[early_stopping_cb,
                                   reduce_lr,
                                   model_checkpoint_cb]
                        )

    training_plot(['loss', 'binary_accuracy'], history,
                  os.path.join(opts['outdir'], 'training_plot.png'))

    # Visualize test images
    image_batch, label_batch = test_dataset.next()
    probabilities = model.predict_on_batch(image_batch).flatten()
    # Apply a sigmoid since our model returns logits
    probabilities = tf.nn.sigmoid(probabilities)
    predictions = tf.where(probabilities < 0.5, 0, 1)
    precision = presision(predictions, probabilities)
    display_batch_of_images((image_batch/255, 
                             label_batch.astype(int)),
                            CLASSES,
                            predictions,
                            precision)

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    # display_confusion_matrix(cmat, score, precision, recall, CLASSES)
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report    

    # because the ImageDataGenerator shuffles the images, but labels not
    # prediction must be done from shuffled dataset
    # normaly it is not necessary, but I shuffled it
    # because visualialization reasons
    test_images = test_dataset.n 
    steps = ceil(test_images / (1.0 * opts['batch_size']))
    test_img, test_label = [], []
    for i in range(steps):
        temp_img, temp_label = test_dataset.next()
        test_img.extend(temp_img)
        test_label.extend(temp_label)

    test_img = np.array(test_img)
    test_label = np.array(test_label)

    test_pred = model.predict(test_img)
    test_prob = tf.nn.sigmoid(test_pred)

    fpr, tpr, threshold = roc_curve(test_label.astype(int), test_prob)
    test_prob_label = tf.where(test_prob < 0.5, 0, 1)
    AUC = auc(fpr, tpr)
    print('AUC:', AUC)
    print(classification_report(test_label.astype(int), test_prob_label))


    ROC_plot(fpr, tpr, AUC)
    results = confusion_matrix(test_label.astype(int), test_prob_label)
    plot_confusion_matrix(results, CLASSES)

    def find_misclassified_img(y_true, y_pred):
        different_positions_FN = []
        different_positions_FP = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                if y_true[i] == 0:
                    different_positions_FP.append(i)
                else:
                    different_positions_FN.append(i)
        return different_positions_FN, different_positions_FP

    diff_pos_FN, diff_pos_FP = find_misclassified_img(test_label.astype(int),
                                                      test_prob_label.numpy()
                                                      )
    misclassified_indx = [*diff_pos_FN, *diff_pos_FP]

    if len(misclassified_indx) > 32:
        print('Number of misclassified images is higher than showed 32 images')
        misclassified_indx = misclassified_indx[:32]
    # labels are lists in list
    # e.g. [[0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1]]
    misclassified_labels = test_prob_label.numpy()[misclassified_indx]
    misclassified_labels = np.hstack(test_prob_label.numpy()[misclassified_indx])

    precision = presision(tf.gather(test_prob_label,
                                    indices=misclassified_indx),
                          tf.gather(test_prob,
                                          indices=misclassified_indx)
                          )

    display_batch_of_images(((test_img[misclassified_indx])/255,
                             test_label[misclassified_indx].astype(int)),
                            CLASSES,
                            misclassified_labels,
                            precision)

    # export the model -- will be written
# =============================================================================
#     export_model(model,
#                  opts['outdir'],
#                  IMAGE_SIZE[0], IMAGE_SIZE[1], IMG_CHANNELS)
# =============================================================================
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

    # Input arguments for training
    parser.add_argument(
        '--job-dir', help='Output directory for results',
        default='D:\\Dropbox (ARG@CS.FEI.VSB)\\Dataset - retiny\\images_from_doctor\\ML\\CLEANING\\clean_medical_imaging_data'
                        )
    parser.add_argument(
        '--input_topdir', help='Directory for the data',
        default='.\\data\\processed\\Classification'
                        )
    parser.add_argument(
        '--num_epochs', help='How many times to iterate over training patterns',
        default=8, type=int)
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
        """)
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help="Starts from checkpoints in output directory")

    # model parameters
    parser.add_argument(
        '--trainable', help='Make model trainable',
        metavar='True or False',
        default=True, type=bool, required=False)
    parser.add_argument(
        '--batch_size', help='Number of records in a batch',
        default=32, type=int)
    parser.add_argument(
        '--l1', help='L1 regularization', default=0., type=float)
    parser.add_argument(
        '--l2', help='L2 regularization', default=0., type=float)
    parser.add_argument(
        '--lrate', help='Adam learning rate', default=0.001, type=float)

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
    strategy = create_strategy(opts['distribute'])  # has to be first/early call in program
    train_and_evaluate(opts, strategy)
