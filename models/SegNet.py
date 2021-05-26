from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D, \
    MaxPooling2D, UpSampling2D,Input, Concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from data_utils import *
import tensorflow as tf
from net_utils import *

class SegNet(object):
    def __init__(self, img_shape, num_of_class, loaded_model = None, path = "saved_models/SegNet_model", actf = 'relu',
        learning_rate = 0.001,  drop_rate = 0.5, do_batch_norm = False, do_drop = False):

        '''
        Arguments :

        img_shape - shape of input image
        actf - activation function for network training
        learning_rate - learning rate for training
        drop_rate - dropout rate
        do_batch_norm - whether to run for batch normalization
        do_drop - whether to run for dropout
        '''

        self.learning_rate = learning_rate
        self.actf = actf
        self.img_shape = img_shape
        self.num_of_class = num_of_class
        self.drop_rate = drop_rate
        self.do_batch_norm = do_batch_norm
        self.do_drop = do_drop
        self.path = path
        self.callbacks = self.create_checkpoints()
        self.model = loaded_model

    
    # encoding block(conv - conv - pool)
    def enc_conv_block_2(self, inputs, feature_maps, filter_size = (3, 3),
                           conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2)):
        conv1 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv2 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool, mask = MaxPoolingWithArgmax2D()(conv2)
        
        return pool, mask
    
    # encoding block(conv - conv - pool)
    def enc_conv_block_4(self, inputs, feature_maps, filter_size = (3, 3),
                           conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2)):
        conv1 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv2 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv3 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv4 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(conv3)
        
        pool, mask = MaxPoolingWithArgmax2D()(conv_4)

        return pool, mask

    
    
    # decoding block(concat - upconv - upconv)
    def dec_conv_block_2(self, inputs, merge_inputs, feature_maps, filter_size = (3, 3), conv_strides = 1,
                           up_conv_strides = (2, 2)):

        #merge = Conv2DTranspose(feature_maps, filter_size, activation = self.actf, strides = up_conv_strides, kernel_initializer = 'he_normal', padding = 'same')(inputs)
        unpool = MaxUnpooling2D()([inputs, merge_inputs])

        conv1 = Conv2D(feature_maps, filter_size, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(unpool)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(self.actf)(conv1)
        
        conv2 = Conv2D(feature_maps, filter_size, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(self.actf)(conv2)

        return conv2
    
    
        # decoding block(concat - upconv - upconv)
    def dec_conv_block_4(self, inputs, merge_inputs, feature_maps, filter_size = (3, 3), conv_strides = 1,
                           up_conv_strides = (2, 2)):

        #merge =  Conv2DTranspose(feature_maps, filter_size, activation = self.actf, strides = up_conv_strides, kernel_initializer = 'he_normal', padding = 'same')(inputs)
        unpool = MaxUnpooling2D()([inputs, merge_inputs])

        conv1 = Conv2D(feature_maps , filter_size , strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(unpool)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(self.actf)(conv1)
        conv2 = Conv2D(feature_maps , filter_size, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(self.actf)(conv2)
        conv3 = Conv2D(feature_maps , filter_size, strides = conv_strides,
               padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(self.actf)(conv3)
        conv4 = Conv2D(feature_maps , filter_size, strides = conv_strides,
               padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(self.actf)(conv4)


        return conv4

    
    # encoder VGG13
    def encoding_path_VGG13(self, inputs):

        enc_conv1, concat1 = self.enc_conv_block_2(inputs, 64)
        enc_conv2, concat2 = self.enc_conv_block_2(enc_conv1, 128)
        enc_conv3, concat3 = self.enc_conv_block_2(enc_conv2, 256)
        enc_conv4, concat4 = self.enc_conv_block_2(enc_conv3, 512)

        return concat1, concat2, concat3, concat4, enc_conv4

    # decoder 
    def decoding_path_VGG13(self, dec_inputs, concat1, concat2, concat3, concat4):

        dec_conv1 = self.dec_conv_block_2(dec_inputs, concat4, 512)
        dec_conv2 = self.dec_conv_block_2(dec_conv1, concat3, 256)
        dec_conv3 = self.dec_conv_block_2(dec_conv2, concat2, 128)
        dec_conv4 = self.dec_conv_block_2(dec_conv3, concat1, 64)

        return dec_conv4
    
    
    # encoder VGG19
    def encoding_path_VGG19(self, inputs):

        enc_conv1, concat1 = self.enc_conv_block_2(inputs, 64)
        enc_conv2, concat2 = self.enc_conv_block_2(enc_conv1, 128)
        enc_conv3, concat3 = self.enc_conv_block_4(enc_conv2, 256)
        enc_conv4, concat4 = self.enc_conv_block_4(enc_conv3, 512)

        return concat1, concat2, concat3, concat4, enc_conv4

    # decoder
    def decoding_path_VGG19(self, dec_inputs, concat1, concat2, concat3, concat4):

        dec_conv1 = self.dec_conv_block_4(dec_inputs, concat4, 512)
        dec_conv2 = self.dec_conv_block_4(dec_conv1, concat3, 256)
        dec_conv3 = self.dec_conv_block_2(dec_conv2, concat2, 128)
        dec_conv4 = self.dec_conv_block_2(dec_conv3, concat1, 64)

        return dec_conv4

    
    # build network
    def build_model_VGG13(self):

        if self.model is None:
            inputs = Input(self.img_shape)

            # Contracting path
            concat1, concat2, concat3, concat4, enc_path = self.encoding_path_VGG13(inputs)

            # middle path 
            mid_path1 = Conv2D(1024, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(enc_path)
            #mid_path1 = Dropout(self.drop_rate)(mid_path1)
            mid_path2 = Conv2D(1024, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(mid_path1)
            #mid_path2 = Dropout(self.drop_rate)(mid_path2)

            # Expanding path
            dec_path = self.decoding_path_VGG13(mid_path2, concat1, concat2, concat3, concat4)
            segmented = Conv2D(self.num_of_class, (1,1), activation ='sigmoid', padding = 'same', kernel_initializer = 'glorot_normal')(dec_path)
            
            self.model = Model(inputs = inputs, outputs = segmented)
            
            
        self.model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = self.learning_rate), metrics = [dice_coef_multilabel, iou_multilabel])

    
        # build network
    def build_model_VGG19(self):

        if self.model is None:
            inputs = Input(self.img_shape)

            # Contracting path
            concat1, concat2, concat3, concat4, enc_path = self.encoding_path_VGG19(inputs)

            mid_path1 = Conv2D(1024, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(enc_path)
            mid_path2 = Conv2D(1024, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(mid_path1)

            mid_path3 = Conv2D(1024, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(mid_path2)

            mid_path4 = Conv2D(1024, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(mid_path3)

            dec_path = self.decoding_path_VGG19(mid_path42, concat1, concat2, concat3, concat4)
            segmented = Conv2D(self.num_of_class, (1,1), activation ='sigmoid', padding = 'same', kernel_initializer = 'glorot_normal')(dec_path)
            
            self.model = Model(inputs = inputs, outputs = segmented)

            
        self.model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = self.learning_rate), metrics = [dice_coef_multilabel, iou_multilabel])

        
        

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

    def save_weights(path):
        self.model.save_weights(path, overwrite=True, save_format=None, options=None)
   
    def set_weights(self, weights):
        self.model.set_weights(weights)
        
    def create_checkpoints(self):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.path,
            save_weights_only=False,
            monitor='val_dice_coef_multilabel',
            mode='max',
            save_best_only=True,
            save_freq="epoch")

        return [model_checkpoint_callback]
    
    # predict test data
    def predict(self, X_test):
        pred_classes = self.model.predict(X_test)

        return pred_classes

    # show architecture
    def show_model(self):
        return print(self.model.summary())

