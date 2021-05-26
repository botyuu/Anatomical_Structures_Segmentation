from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D, \
    MaxPooling2D, UpSampling2D,Input, Concatenate, Conv2DTranspose, ZeroPadding2D, Add, AveragePooling2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from data_utils import *
import tensorflow as tf
from net_utils import *

class Resnet50v2_Unet(object):
    def __init__(self, img_shape, num_of_class, loaded_model = None, path = "saved_models/Resnet50_UNet_model_2v", actf = 'relu',
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

    def decoder(self, x, f4, f3, f2, f1, l1_skip_conn = True):
        
        #[f1, f2, f3, f4, f5] = levels
        x = Conv2DTranspose(1024, 3, activation = self.actf, strides = 2, kernel_initializer = 'he_normal', padding = 'same')(x)
        x = Concatenate()([x, f4])
        x = (Conv2D(1024, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        x = (Conv2D(1024, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        
        
        x = Conv2DTranspose(512, 3, activation = self.actf, strides = 2, kernel_initializer = 'he_normal', padding = 'same')(x)
        x = Concatenate()([x, f3])
        x = (Conv2D(512, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        x = (Conv2D(512, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        
        x = Conv2DTranspose(256, 3, activation = self.actf, strides = 2, kernel_initializer = 'he_normal', padding = 'same')(x)
        x = Concatenate()([x, f2])
        x = (Conv2D(256, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        x = (Conv2D(256, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        
        x = Conv2DTranspose(128, 3, activation = self.actf, strides = 2, kernel_initializer = 'he_normal', padding = 'same')(x)
        #x = Concatenate()([x, f3])
        x = (Conv2D(128, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        x = (Conv2D(128, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
      
        #x = Conv2DTranspose(128, 3, activation = self.actf, strides = 2, kernel_initializer = 'he_normal', padding = 'same')(x)
        #x = Concatenate()([x, f3])
        x = (Conv2D(64, (1, 1), strides = 2, padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)        
        x = Concatenate()([x, f1])
        x = (Conv2D(64, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        x = (Conv2D(64, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        
        x = Conv2DTranspose(32, 3, activation = self.actf, strides = 2, kernel_initializer = 'he_normal', padding = 'same')(x)
        x = (Conv2D(32, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        x = (Conv2D(32, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        
        
        
        x = Conv2DTranspose(16, 3, activation = self.actf, strides = 2, kernel_initializer = 'he_normal', padding = 'same')(x)
        x = (Conv2D(16, (1, 1), padding='valid', activation='relu'))(x)
        x = (BatchNormalization())(x)
        
        
        return x    
    
    def create_conv_layer(self, x, feature_map, strides, kernel_size):
        x = Conv2D(feature_map, kernel_size, strides=strides)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def create_conv_layers(self, x, feature_map, feature_map2, strides = 1, strides2 = 1, kernel_size=1):
        x = self.create_conv_layer(x, feature_map, strides, kernel_size)
        x = self.create_conv_layer(x, feature_map, strides2, kernel_size)
        x = Conv2D(feature_map2, kernel_size, strides=strides2)(x)
        x = BatchNormalization()(x)
        return x
    
    def add_layers_with_identity(self, x, feature_map, feature_map2, strides1 = 1, strides2 = 1, kernel_size=1):
        x2 = self.create_conv_layers(x, feature_map, feature_map2, strides1, strides2, kernel_size)
        x = Conv2D(feature_map2, kernel_size, strides1)(x)
        x = BatchNormalization()(x)
        x = Add()([x, x2])
        x = Activation('relu')(x)
        return x
    

    def add_layers(self, x, feature_map, feature_map2, strides = 1, kernel_size=1):
        x2 = self.create_conv_layers(x, feature_map, feature_map2, strides, strides, kernel_size)
        x = Add()([x, x2])
        x = Activation('relu')(x)
        return x
    
    def encoder(self):
        inputs = Input(self.img_shape)

            
        x = ZeroPadding2D((3, 3))(inputs)
        x = self.create_conv_layer(x, 64, strides = 2, kernel_size = 7)
        x = ZeroPadding2D((1, 1))(x)
        
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        
        f1 = x
        
        x = self.add_layers_with_identity(x, 64, 256)
        x = self.add_layers(x, 64, 256)
        x= self.add_layers(x, 64, 256)
        
        f2 = x
        
        x = self.add_layers_with_identity(x, 128, 512, 2, 1)
        x = self.add_layers(x, 128, 512)
        x = self.add_layers(x, 128, 512)
        x = self.add_layers(x, 128, 512)
        
        f3 = x
        
        x = self.add_layers_with_identity(x, 256, 1024, 2, 1)
        x = self.add_layers(x, 256, 1024)
        x = self.add_layers(x, 256, 1024)
        x = self.add_layers(x, 256, 1024)
        x = self.add_layers(x, 256, 1024)
        x = self.add_layers(x, 256, 1024)
        
        f4 = x

        x = self.add_layers_with_identity(x, 512, 2048, 2)
        x = self.add_layers(x, 512, 2048)
        x = self.add_layers(x, 512, 2048)

       
        return inputs, x, f4, f3, f2, f1


        # build network
    def build_model(self):

        if self.model is None:
            
            inputs, x, f4, f3, f2, f1 = self.encoder()
            x = self.decoder(x, f4, f3, f2, f1)
            #x = self.decoder(x, levels)
            #x = (Reshape(((self.img_shape[0], self.img_shape[1], self.num_of_class))))(x)
            segmented = Conv2D(self.num_of_class, (1,1), activation ='sigmoid', padding = 'same', kernel_initializer = 'glorot_normal')(x)

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

    # show u-net architecture
    def show_model(self):
        return print(self.model.summary())

