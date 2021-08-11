import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow.keras.backend as K
from sklearn import model_selection, metrics
from sklearn.metrics import classification_report
import math


# data evalutate function
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

def dice_coef_multilabel(y_true, y_pred, numLabels = 1):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average

def iou_multilabel(y_true, y_pred, numLabels = 1):
    iou_sum=0
    for index in range(numLabels):
        iou_sum += iou(y_true[:,:,:,index], y_pred[:,:,:,index])
    return iou_sum/numLabels # taking average

def iou(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

# data pre-processing function
# standardization of data
def data_std(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean)/(std + 1e-7)
    X_test = (X_test - mean)/(std + 1e-7)

    return X_train, X_test

# plot history for accuracy
def plot_dice(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['dice_coef_multilabel'])
    plt.plot(history['val_dice_coef_multilabel'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Dice_coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc=0)


def plot_loss_dice(history):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, y)
    axs[0, 1].plot(history['val_dice_coef_multilabel'])
    axs[0, 1].plot(history['dice_coef_multilabel'])
    axs[0, 1].ylabel('Dice_coefficient')
    axs[0, 1].xlabel('Epoch')
    axs[0, 1].legend(['Train', 'Val'], loc=0)
    axs[1, 1].plot(history['val_loss'])
    axs[1, 1].plot(history['loss'])
    axs[1, 1].ylabel('Loss')
    axs[1, 1].xlabel('Epoch')
    axs[1, 1].legend(['Train', 'Val'], loc=0)

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

# plot history for loss
def plot_loss(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)

def combine_mask(mask):
    #SPINAL CORD
    tmp_LL = np.zeros((mask.shape[0], mask.shape[1], 3), dtype = np.uint8)
    tmp_LL[:, :, 0] = (mask[:, :, 0]*255).astype(np.uint8)
    tmp_LL[:, :, 1] = tmp_LL[:, :, 0].astype(np.uint8)
    tmp_LL[:, :, 2] = tmp_LL[:, :, 0].astype(np.uint8)
    #LEFT LUNG
    tmp_LR = np.zeros((mask.shape[0], mask.shape[1], 3), dtype = np.uint8)
    tmp_LR[:, :, 0] = (mask[:, :, 1]).astype(np.uint8)
    tmp_LR[:, :, 1] = (tmp_LR[:, :, 0]*179).astype(np.uint8)
    tmp_LR[:, :, 2] = (tmp_LR[:, :, 0]*255).astype(np.uint8)
    tmp_LR[:, :, 0] = (tmp_LR[:, :, 0]*0).astype(np.uint8)
    #RIGHT LUNG
    tmp_E = np.zeros((mask.shape[0], mask.shape[1], 3), dtype = np.uint8)
    tmp_E[:, :, 0] = (mask[:, :, 2]).astype(np.uint8)
    tmp_E[:, :, 1] = (tmp_E[:, :, 0]*105)
    tmp_E[:, :, 2] = (tmp_E[:, :, 0]*182)
    tmp_E[:, :, 0] = (tmp_E[:, :, 0]*93)
    #HEART
    tmp_H = np.zeros((mask.shape[0], mask.shape[1], 3), dtype = np.uint8)
    tmp_H[:, :, 0] = (mask[:, :, 3]*255).astype(np.uint8)
    #ESOPHAGUS
    tmp_S = np.zeros((mask.shape[0], mask.shape[1], 3), dtype = np.uint8)
    tmp_S[:, :, 0] = (mask[:, :, 4]).astype(np.uint8)
    tmp_S[:, :, 1] = (tmp_S[:, :, 0]*153).astype(np.uint8)
    tmp_S[:, :, 2] = (tmp_S[:, :, 0]*153).astype(np.uint8)
    tmp_S[:, :, 0] = (tmp_S[:, :, 0]*255).astype(np.uint8)
    
    return tmp_S+tmp_H+tmp_E+tmp_LR+tmp_LL
    
def show_imgs(show_num, imgs, masks, pred_masks):
    fig, ax = plt.subplots(show_num, 3, figsize=(15, 50))
    for i in range(show_num):
        ax[i, 0].imshow(imgs[i].squeeze(), cmap=plt.cm.bone)
        ax[i, 1].imshow(combine_mask(masks[i].squeeze()))
        ax[i, 2].imshow(combine_mask(pred_masks[i].squeeze()))

    
    