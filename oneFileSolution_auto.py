# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:12:30 2023

@author: has081

One file solution autoencoder
"""

import math, re, os, sys
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
np.set_printoptions(threshold=np.inf)

from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    LeakyReLU,
    Activation,
    concatenate,
    Flatten,
    Reshape,
)
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import statistics
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = (256, 256) 
IMG_CHANNELS = 3
BATCH_SIZE = 64 
EPOCHS = 400

# Preprocessing parameters
RESCALE = 1.0 / 255
SHAPE = IMAGE_SIZE
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN

# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)


def batch_to_numpy_images_and_labels(data):
    images, labels = data
    # print('Zde ',labels  )
    if type(labels) == np.ndarray:
        # print(labels)
        return images, labels
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    # binary string in this case, these are image ID strings
    if numpy_labels.dtype == object: 
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    # print(label, correct_label)
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', 
                                u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_img(image, title, subplot, red=False, titlesize=16):
    
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), 
                  color='red' if red else 'black', fontdict={'verticalalignment':'center'}, 
                  pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images(databatch, predictions=None, precision=None):
    """Function works with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows

    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols, FIGSIZE))

    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
            # print(title)
        if precision is not None:
            title = title +'-'+ str(np.round(precision[i],3))
        # magic formula tested to work from 1x1 to 10x10 images
        dynamic_titlesize = FIGSIZE*SPACING/max(rows, cols)*40+3
        subplot = display_one_img(
            image, title, subplot, not correct, titlesize=dynamic_titlesize)

    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


def display_side_by_side(images1, images2, labels, save=False):
    # Compute the number of rows for the subplots
    num_images = len(images1)
    num_cols = 2

    # Create a figure with the subplots
    fig, axes = plt.subplots(num_images, 2, figsize=(num_cols * 2, num_images * 2))

    # Display the images from the first array on the left side
    for i in range(num_images):
        row = i 
        axes[row, 0].imshow(images1[i])
        axes[row, 0].set_title(f'{CLASSES[int(labels[i])]}')
        axes[row, 0].axis("off")
        axes[row, 0].set_adjustable("box")
        
        axes[row, 1 ].imshow(images2[i])
        axes[row, 1 ].set_title(f'MSE: {mse(images1[i], images2[i]):.3f}, SSIM: {SSIMLoss(images1[i], images2[i]).numpy():.3f}')
        axes[row, 1 ].axis("off")
        axes[row, 1 ].set_adjustable("box") 
        #row.set_title(' title', fontweight='semibold')

    # Show the figure
    fig.suptitle('Input : Reconstructed')
    if save:
        plt.savefig('result.png')
    plt.show()
    
    
def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    
def display_training_curves(training, validation, title, subplot, zoom_pcent=None, ylim=None):
    # zoom_pcent: X autoscales y axis for the last X% of data points
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    if zoom_pcent is not None:
        ylen = len(training)*(100-zoom_pcent)//100
        ymin = min([min(training[ylen:]), min(validation[ylen:])])
        ymax = max([max(training[ylen:]), max(validation[ylen:])])
        ax.set_ylim([ymin-(ymax-ymin)/20, ymax+(ymax-ymin)/20])
    if ylim is not None:
        ymin = ylim[0]
        ymax = ylim[1]
        ax.set_ylim([ymin-(ymax-ymin)/20, ymax+(ymax-ymin)/20])
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
    
    
def plot_label_clusters(encoder, data, labels, vae=True):
    # display a 2D plot of the classes in the latent space
    if vae:
      z_mean, _, _ = encoder.predict(data)
    else:
      z_mean = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    for label in range(2): # classification, 2 digits
        cx = np.mean(z_mean[labels == label, 0])
        cy = np.mean(z_mean[labels == label, 1])
        plt.text(cx, cy, str(label), color="white", fontsize=25, fontweight="bold")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

    
datagen = ImageDataGenerator(rotation_range=70,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale= 1 / 255.0)


img = load_img('data/processed/Processed1/train/Good/210301_Supolikova_Nina_R_04-15-2021_10-25-25_1.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, x, x)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, x, x)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# =============================================================================
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='data', save_prefix='from_datagen', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely
# =============================================================================
  
# load and iterate training dataset
# opts['input_topdir'], 'valid' + opts['pattern']
train_dataset = datagen.flow_from_directory("data/processed/Processed1_autoencoder/train", 
                                       class_mode='input', 
                                       color_mode="rgb",
                                       target_size=IMAGE_SIZE,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True)
# load and iterate validation dataset
validation_dataset = datagen.flow_from_directory("data/processed/Processed1_autoencoder/val", 
                                       class_mode='input', 
                                       color_mode="rgb",
                                       target_size=IMAGE_SIZE,
                                       # batch_size=opts['batch_size']
                                       batch_size=BATCH_SIZE,
                                       shuffle=True)
# load and iterate test dataset
test_datagen = ImageDataGenerator(rescale= 1 / 255.0)
test_dataset = test_datagen.flow_from_directory("data/processed/Processed1/test/",  
                                       color_mode="rgb",
                                       class_mode='binary', 
                                       target_size=IMAGE_SIZE,
                                       batch_size=32,
                                       shuffle=True)

class_names_train = train_dataset.class_indices
class_names_val = validation_dataset.class_indices
# class_names_val == class_names_train # wow there are the same
class_names_test = test_dataset.class_indices
CLASSES = list (class_names_test.keys())

# images count per category
# from collections import Counter
# counter = Counter(train_dataset.classes)




def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    img_dim = (*SHAPE, channels)

    # input
    input_img = Input(shape=img_dim)

    # encoder
    encoding_dim = 256  # 64
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(
        input_img
    )
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Flatten()(x)
    x = Dense(encoding_dim, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    # encoded = x

    # decoder
    x = Reshape((4, 4, encoding_dim // 16))(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(
        img_dim[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    decoded = x
    # model
    autoencoder = Model(input_img, decoded)
    return autoencoder




autoencoder = build_model('rgb')
# how many layers are in the model
print("Number of layers in the model: ", len(autoencoder.layers))
# print(len(autoencoder.trainable_variables))
# autoencoder.summary()


import scipy.ndimage as nd
import scipy.ndimage.filters as filters
from keras import losses
import tensorflow as tf
# https://github.com/ilopezfr/image-superres/blob/master/Image_Super_Resolution_using_Autoencoders.ipynb

def hfenn_loss(ori, res):
    '''
    HFENN-based loss
    ori, res - batched images with 3 channels
    See metrics.hfenn
    '''
    fnorm = 0.325 # norm of l_o_g operator, estimated numerically
    sigma = 1.5 # parameter from HFEN metric
    truncate = 4 # default parameter from filters.gaussian_laplace
    wradius = int(truncate * sigma + 0.5)
    eye = np.zeros((2*wradius+1, 2*wradius+1), dtype=np.float32)
    eye[wradius, wradius] = 1.
    ker_mat = filters.gaussian_laplace(eye, sigma)
    with tf.name_scope('hfenn_loss'):
        chan = 3
        ker = tf.constant(np.tile(ker_mat[:, :, None, None], (1, 1, chan, 1)))
        filtered = tf.nn.depthwise_conv2d(ori - res, ker, [1, 1, 1, 1], 'VALID')
        loss = tf.reduce_mean(tf.square(filtered))
        loss = loss / (fnorm**2)
    return loss
  

def ae_loss(input_img, decoder):
    mse = losses.mean_squared_error(input_img, decoder) # MSE
    weight = 10.0 # weight
    return mse + weight * hfenn_loss(input_img, decoder) # MSE + weight * HFENN



LR = 0.0001
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss=ae_loss,
              run_eagerly=True)


autoencoder.summary()




loss0 = autoencoder.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))


history = autoencoder.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)



loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('MSE')
#plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



""" ANOMALY DETECTION """



# 1. Reconstruct the data using our trainined autoencoder model.
x_test_recon = autoencoder.predict(test_dataset)

# autoencoder.compile(optimizer=optimizer, loss=SSIMLoss) # possibility
def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA - imageB) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


# https://pyimagesearch.com/2014/09/15/python-compare-two-images/
test_images = test_dataset.n 
steps = math.ceil(test_images / (1.0 * 32)) 
test_img , test_label = [] , []
for i in range(steps):
    t , l = test_dataset.next()
    test_img.extend(t) 
    test_label.extend(l)
    
test_img = np.array(test_img)
test_label = np.array(test_label)
# the reconstruction score is the mean of the reconstruction errors (relatively high scores are anomalous)
mse_test = [mse(test_img[i],x_test_recon[i])  for i in range(test_images)]
SSIM_test = [SSIMLoss(test_img[i],x_test_recon[i])  for i in range(test_images)]

# store the reconstruction data in a Pandas dataframe
anomaly_data = pd.DataFrame({'recon_score_mse':mse_test,
                             'recon_score_ssim':[x.numpy() for x in SSIM_test],
                             'label':test_label.astype(int)})

# if our reconstruction scores our normally distributed we can use their statistics
anomaly_data.describe()

# plotting the density will give us an idea of how the reconstruction scores are distributed
plt.xlabel('Reconstruction Score')
anomaly_data['recon_score_mse'].plot.hist(bins=200, range=[.05, .9])

labels = anomaly_data.label.unique()
plt.hist([anomaly_data.loc[anomaly_data.label == x, 'recon_score_mse'] for x in labels], label=labels)
plt.title('recon_score_mse')
plt.legend(CLASSES)
#plt.show()

plt.hist([anomaly_data.loc[anomaly_data.label == x, 'recon_score_ssim'] for x in labels], label=labels)
plt.title('recon_score_ssim')
plt.legend(CLASSES)
plt.show()


anomaly_data.groupby(by='label').describe()


from sklearn.metrics import roc_curve, auc, confusion_matrix
fpr, tpr, thresholds = roc_curve(test_label.astype(int), mse_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='lime', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



# There are number of ways to set thresholds. We give some examples below and we encourage you to experiment by setting your own thresholds and see how they affect the model.
# We can pick the threshold based on maximizing the true positive rate (tpr) 
# and minimizing the false positive rate (fpr)
optimal_threshold_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_idx]
print(optimal_threshold)

# Or we assume our reconstructions are normally distributed and label anomalies as those
# that are a number of standard deviations away from the mean
recon_mean = np.mean(mse_test)
recon_stddev = np.std(mse_test)

stats_threshold = recon_mean + 5*recon_stddev
print(stats_threshold)


thresh = optimal_threshold
#thresh = stats_threshold



print(thresh)

pred_labels = (mse_test > thresh).astype(int)

results = confusion_matrix(test_label.astype(int), pred_labels) 


print ('Confusion Matrix: ')

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


plot_confusion_matrix(results, ['Normal','Anomaly'])




def presision(predictions, probabilities):
    pre = predictions.numpy()
    pro = probabilities.numpy()
    return 1 - np.abs(pre - pro)
    




### test images
image_batch, label_batch = test_dataset.next()
prediction = autoencoder.predict_on_batch(image_batch)
display_side_by_side(image_batch[:5], prediction[:5],label_batch[:5])


"""
def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    img_dim = (*SHAPE, channels)

    # input
    input_img = Input(shape=img_dim)

    # encoder
    encoding_dim = 256  # 64
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(
        input_img
    )
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Flatten()(x)
    x = Dense(encoding_dim, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    # encoded = x

    # decoder
    x = Reshape((4, 4, encoding_dim // 16))(x)
    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(
        img_dim[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    decoded = x
    # model
    autoencoder = Model(input_img, decoded)
    return autoencoder

"""
















