from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D, \
    MaxPooling2D, UpSampling2D,Input, Concatenate, Conv2DTranspose, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from data_utils import *
import tensorflow as tf
from net_utils import *

class TernausNet(object):
    def __init__(self, img_shape, num_of_class, loaded_model = None, path = "saved_models/TernausNet_model", actf = 'relu',
        learning_rate = 0.001, normalization = False, maxPooling = True):

        '''
        Arguments :

        img_shape - shape of input image (64, 64, 1)
        actf - activation function for network training
        learning_rate - learning rate for training
        '''

        self.learning_rate = learning_rate
        self.actf = actf
        self.img_shape = img_shape
        self.num_of_class = num_of_class
        self.path = path
        self.callbacks = self.create_checkpoints()
        self.model = loaded_model
        self.normalization = normalization
        self.maxPooling = maxPooling

        self.build_model()
    # encoding block
    def enc_conv_block(self, x, layers, feature_maps, filter_size = (3, 3),
                           conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2)):
        for i in range(layers):
            x = Conv2D(feature_maps , filter_size, strides = conv_strides, 
                       padding = 'same', kernel_initializer = 'he_normal')(x)
            if self.normalization:
                x = BatchNormalization()(x)
            x = Activation(self.actf)(x)
        if self.maxPooling == False:
            pool = AveragePooling2D(pooling_filter_size, strides = pooling_strides)(x)
        else:
            pool = MaxPooling2D(pooling_filter_size, strides = pooling_strides)(x)

        return pool, x

    # decoding block
    def dec_conv_block(self, inputs, merge_inputs, layers, feature_maps, trans_feature_maps, filter_size = (3, 3), conv_strides = 1,
                           up_conv_strides = (2, 2)):

        merge = Concatenate(axis = 3)([Conv2DTranspose(trans_feature_maps, filter_size,
                                                       activation = self.actf, strides = up_conv_strides, kernel_initializer = 'he_normal',
                                                       padding = 'same')(inputs), merge_inputs])
        x = merge
        for i in range(layers):
            x = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                               padding = 'same', kernel_initializer = 'he_normal')(x)
            if self.normalization:
                x = BatchNormalization()(x)
            x = Activation(self.actf)(x)
        return x

    # encoder
    def encoding_path(self, inputs):

        enc_conv1, concat1 = self.enc_conv_block(inputs, 1, 64)
        enc_conv2, concat2 = self.enc_conv_block(enc_conv1, 1, 128)
        enc_conv3, concat3 = self.enc_conv_block(enc_conv2, 2, 256)
        enc_conv4, concat4 = self.enc_conv_block(enc_conv3, 2, 512)
        enc_conv5, concat5 = self.enc_conv_block(enc_conv4, 2, 512)

        return concat1, concat2, concat3, concat4, concat5, enc_conv5

    # decoder
    # In decoding_path, the filter outputs is half of the previos
    def decoding_path(self, dec_inputs, concat1, concat2, concat3, concat4, concat5):

        dec_conv1 = self.dec_conv_block(dec_inputs, concat5, 1, 512, 256)
        dec_conv2 = self.dec_conv_block(dec_conv1, concat4, 1, 512, 256)
        dec_conv3 = self.dec_conv_block(dec_conv2, concat3, 1, 256, 128)
        dec_conv4 = self.dec_conv_block(dec_conv3, concat2, 1, 128, 64)
        dec_conv5 = self.dec_conv_block(dec_conv4, concat1, 1, 64, 32)

        return dec_conv5
    
    # build network
    def build_model(self):
        
        
        if self.model is None:
            inputs = Input(self.img_shape)

            # Contracting path
            concat1, concat2, concat3, concat4, concat5, enc_path = self.encoding_path(inputs)

            # center
            center = Conv2D(512, (3,3), padding = 'same', kernel_initializer = 'he_normal')(enc_path)
            if self.normalization:
                center = BatchNormalization()(center)
            center = Activation(self.actf)(center)
            # Expanding path
            dec_path = self.decoding_path(center, concat1, concat2, concat3, concat4, concat5)
            segmented = Conv2D(self.num_of_class, (1,1), activation ='sigmoid', padding = 'same', kernel_initializer = 'glorot_normal')(dec_path)

            self.model = Model(inputs = inputs, outputs = segmented)
        self.model.compile(optimizer = Adam(learning_rate = self.learning_rate),
                          loss = 'binary_crossentropy', metrics = [dice_coef_multilabel, iou_multilabel])

    
    def create_checkpoints(self):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.path,
            save_weights_only=False,
            monitor='val_dice_coef_multilabel',
            mode='max',
            save_best_only=True,
            save_freq="epoch")

        return [model_checkpoint_callback]
    
    # train model
    def train(self, X_train, Y_train, X_valid, Y_valid, epoch = 10, batch_size = 32):

        self.history = self.model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid),
                                          epochs = epoch, batch_size = batch_size, callbacks=self.callbacks)
        return self.history

    # train model with augumantation
    def train_aug(self, generator, X_train, Y_train, X_valid, Y_valid, epoch = 10, batch_size = 32):

        self.history = self.model.fit_generator(generator.flow(X_train, Y_train, batch_size = batch_size), 
                    validation_data=(X_valid, Y_valid),  epochs = epoch, callbacks=self.callbacks)
        return self.history
    
    # predict test data
    def predict(self, X_test):
        pred_classes = self.model.predict(X_test)

        return pred_classes

    # show architecture
    def show_model(self):
        return print(self.model.summary())
