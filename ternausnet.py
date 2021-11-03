from models.TernausNet import *

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import time

path = "/media/bot/DANE/MED_DATASETS/NARZ_ZAGRO2/manifest-1557326747206/LCTSC"
no_epoch = 100
no_batch_size = 8

images = np.load(os.path.join(path, "imgs_256_full.npy"))
labels = np.load(os.path.join(path, "masks_256_full.npy"))


N = images.shape[0]
W = images.shape[1]
H = images.shape[2]
C = labels.shape[3]

x_train, x_test_valid, y_train, y_test_valid = train_test_split(images, labels, test_size=0.4, random_state=4)
x_valid, x_test, y_valid, y_test = train_test_split(x_test_valid, y_test_valid, test_size=0.5, random_state=4)

print("Size of train set: ", x_train.shape, y_train.shape)

seg_model = TernausNet(img_shape = x_train[0].shape, num_of_class = 5, learning_rate = 2e-4, path = "saved_models/ternausnet_full_imgs")
seg_model.show_model()

history = seg_model.train(x_train, y_train, x_valid, y_valid, epoch = no_epoch, batch_size = no_batch_size)