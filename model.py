# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:00:16 2019

@author: Aidan
"""

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Set some parameters
im_width = 160
im_height = 160
path_train = "C:/Users/Aidan/dstl-satellite-imagery-feature-detection/"
path_test = "C:/Users/Aidan/dstl-satellite-imagery-feature-detection/"
num_classes = 11
image_shape = (160, 160)


# Get and resize train images and masks
def get_data(data_folder):
    data = np.concatenate((np.load(os.path.join(data_folder, "data/train_data_0a.npy")), np.load(os.path.join(data_folder, "data/train_data_0b.npy")), np.load(os.path.join(data_folder, "data/train_data_0c.npy")), np.load(os.path.join(data_folder, "data/train_data_0d.npy"))), axis=0)
    labels = np.concatenate((np.load(os.path.join(data_folder, "labels/train_labels_0a.npy")), np.load(os.path.join("labels/train_labels_0b.npy")), np.load(os.path.join(data_folder, "labels/train_labels_0c.npy")), np.load(os.path.join(data_folder, "labels/train_labels_0d.npy"))), axis=0)
    d#ata = np.load(os.path.join(data_folder, "data/train_data_0.npy"))
    l#abels = np.load(os.path.join(data_folder, "labels/train_labels_0.npy"))
    return data, labels


X, y = get_data(path_train)

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0,
                                                      random_state=2018)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0,
                                                    random_state=2018)

# Check if training data looks alright
ix = random.randint(0, len(X_train))

fig, ax = plt.subplots(12, 1, figsize=(10, 20))

ax[0].imshow(X_train[ix], interpolation="bilinear")
ax[0].set_title("Picture")

for i in range(1, 12):
    ax[i].imshow(y_train[ix, ..., i - 1], interpolation="bilinear", cmap="gray")
    ax[i].set_title("Label " + str(i))

plt.show()


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layers
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3,
                      batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3,
                      batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3,
                      batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3,
                      batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3,
                      batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2),
                         padding="same")(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3,
                      batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2),
                         padding="same")(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3,
                      batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2),
                         padding="same")(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3,
                      batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2),
                         padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3,
                      batchnorm=batchnorm)

    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


input_img = Input((im_height, im_width, 3), name="img")
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="categorical_crossentropy",
              metrics=["accuracy"])
print(model.summary())

callbacks = [EarlyStopping(patience=10, verbose=True),
             ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001,
                               verbose=True),
             ModelCheckpoint("segmentation-model.h5", verbose=True,
                             save_best_only=True, save_weights_only=True)]

results = model.fit(X_train, y_train, batch_size=4, epochs=20,
                    callbacks=callbacks, validation_data=(X_valid, y_valid))

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()
plt.show()

# Load best model
model.save_weights("C:/Users/Aidan/dstl-satellite-imagery-feature-detection/segmentation-model.h5")
model.load_weights("C:/Users/Aidan/dstl-satellite-imagery-feature-detection/segmentation-model.h5")

# Predict on train, val, and test
preds_train = model.predict(X_train, verbose=True)
preds_val = model.predict(X_test, verbose=True)

# Threshold predictions
preds_train_t = (preds_train > 0.1).astype(np.uint8)
preds_val_t = (preds_val > 0.1).astype(np.uint8)


def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))
        print("ix:", ix)

    fig, ax = plt.subplots(7, 1, figsize=(10, 20))

    ax[0].imshow(X_train[ix], interpolation="bilinear")
    ax[0].set_title("Picture")

    for i in range(3, 6):
        ax[2 * i - 5].imshow(y_train[ix, ..., i], interpolation="bilinear", cmap="gray")
        ax[2 * i - 5].set_title("True Label")

        ax[2 * i - 4].imshow(preds[ix, ..., i], interpolation="bilinear", cmap="gray")
        ax[2 * i - 4].set_title("Predicted Label")

    plt.show()

    # buildings = red
    true_buildings_overlay = (y[ix, ..., 1] > 0).reshape(im_height, im_width, 1)
    true_buildings_overlay_rgba = np.concatenate((true_buildings_overlay, np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay * 0.5), axis=2)
    # structures = gray
    true_structures_overlay = (y[ix, ..., 2] > 0).reshape(im_height, im_width, 1)
    true_structures_overlay_rgba = np.concatenate((true_structures_overlay / 4, true_structures_overlay / 4, true_structures_overlay / 4, true_structures_overlay * 0.5), axis=2)
    # road = black
    true_road_overlay = (y[ix, ..., 3] > 0).reshape(im_height, im_width, 1)
    true_road_overlay_rgba = np.concatenate((np.zeros(true_road_overlay.shape), np.zeros(true_road_overlay.shape), np.zeros(true_road_overlay.shape), true_road_overlay * 0.5), axis=2)
    # track = brown
    true_track_overlay = (y[ix, ..., 4] > 0).reshape(im_height, im_width, 1)
    true_track_overlay_rgba = np.concatenate((true_track_overlay * 2 / 3, true_track_overlay / 3, np.zeros(true_track_overlay.shape), true_track_overlay * 0.5), axis=2)
    # trees = green
    true_trees_overlay = (y[ix, ..., 5] > 0).reshape(im_height, im_width, 1)
    true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay * 0.5), axis=2)
    # crops = yellow
    true_crops_overlay = (y[ix, ..., 6] > 0).reshape(im_height, im_width, 1)
    true_crops_overlay_rgba = np.concatenate((true_crops_overlay, true_crops_overlay, np.zeros(true_crops_overlay.shape), true_crops_overlay * 0.5), axis=2)
    # waterways = cyan
    true_waterway_overlay = (y[ix, ..., 7] > 0).reshape(im_height, im_width, 1)
    true_waterway_overlay_rgba = np.concatenate((np.zeros(true_waterway_overlay.shape), true_waterway_overlay, true_waterway_overlay, true_waterway_overlay * 0.5), axis=2)
    # standing_water = blue
    true_standing_water_overlay = (y[ix, ..., 8] > 0).reshape(im_height, im_width, 1)
    true_standing_water_overlay_rgba = np.concatenate((true_standing_water_overlay, np.zeros(true_standing_water_overlay.shape), np.zeros(true_standing_water_overlay.shape), true_standing_water_overlay * 0.5), axis=2)
    # vehicles_large = magenta
    true_vehicle_large_overlay = (y[ix, ..., 9] > 0).reshape(im_height, im_width, 1)
    true_vehicle_large_overlay_rgba = np.concatenate((true_vehicle_large_overlay, np.zeros(true_vehicle_large_overlay.shape), true_vehicle_large_overlay, true_vehicle_large_overlay * 0.5), axis=2)
    # vehicle_small = orange
    true_vehicle_small_overlay = (y[ix, ..., 10] > 0).reshape(im_height, im_width, 1)
    true_vehicle_small_overlay_rgba = np.concatenate((true_vehicle_small_overlay, true_vehicle_small_overlay / 2, np.zeros(true_vehicle_small_overlay.shape), true_vehicle_small_overlay * 0.5), axis=2)
    # background = white
    true_background_overlay = (y[ix, ..., 0] > 0).reshape(im_height, im_width, 1)
    true_background_overlay_rgba = np.concatenate((true_background_overlay, true_background_overlay, true_background_overlay, true_background_overlay * 0.5), axis=2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(X[ix], interpolation="bilinear")
    ax.imshow(true_background_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_buildings_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_structures_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_road_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_track_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_trees_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_crops_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_waterway_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_standing_water_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_vehicle_large_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_vehicle_small_overlay_rgba, interpolation="bilinear")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    # buildings = red
    buildings_overlay = (binary_preds[ix, ..., 1] > 0).reshape(im_height, im_width, 1)
    buildings_overlay_rgba = np.concatenate((buildings_overlay, np.zeros(buildings_overlay.shape), np.zeros(buildings_overlay.shape), buildings_overlay * 0.5), axis=2)
    # structures = gray
    structures_overlay = (binary_preds[ix, ..., 2] > 0).reshape(im_height, im_width, 1)
    structures_overlay_rgba = np.concatenate((structures_overlay / 4, structures_overlay / 4, structures_overlay / 4, structures_overlay * 0.5), axis=2)
    # road = black
    road_overlay = (binary_preds[ix, ..., 3] > 0).reshape(im_height, im_width, 1)
    road_overlay_rgba = np.concatenate((np.zeros(road_overlay.shape), np.zeros(road_overlay.shape), np.zeros(road_overlay.shape), road_overlay * 0.5), axis=2)
    # track = brown
    track_overlay = (binary_preds[ix, ..., 4] > 0).reshape(im_height, im_width, 1)
    track_overlay_rgba = np.concatenate((track_overlay * 2 / 3, track_overlay / 3, np.zeros(track_overlay.shape), track_overlay * 0.5), axis=2)
    # trees = green
    trees_overlay = (binary_preds[ix, ..., 5] > 0).reshape(im_height, im_width, 1)
    trees_overlay_rgba = np.concatenate((np.zeros(trees_overlay.shape), trees_overlay, np.zeros(trees_overlay.shape), trees_overlay * 0.5), axis=2)
    # crops = yellow
    crops_overlay = (binary_preds[ix, ..., 6] > 0).reshape(im_height, im_width, 1)
    crops_overlay_rgba = np.concatenate((crops_overlay, crops_overlay, np.zeros(crops_overlay.shape), crops_overlay * 0.5), axis=2)
    # waterways = cyan
    waterway_overlay = (binary_preds[ix, ..., 7] > 0).reshape(im_height, im_width, 1)
    waterway_overlay_rgba = np.concatenate((np.zeros(waterway_overlay.shape), waterway_overlay, waterway_overlay, waterway_overlay * 0.5), axis=2)
    # standing_water = blue
    standing_water_overlay = (binary_preds[ix, ..., 8] > 0).reshape(im_height, im_width, 1)
    standing_water_overlay_rgba = np.concatenate((standing_water_overlay, np.zeros(standing_water_overlay.shape), np.zeros(standing_water_overlay.shape), standing_water_overlay * 0.5), axis=2)
    # vehicles_large = magenta
    vehicle_large_overlay = (binary_preds[ix, ..., 9] > 0).reshape(im_height, im_width, 1)
    vehicle_large_overlay_rgba = np.concatenate((vehicle_large_overlay, np.zeros(vehicle_large_overlay.shape), vehicle_large_overlay, vehicle_large_overlay * 0.5), axis=2)
    # vehicle_small = orange
    vehicle_small_overlay = (binary_preds[ix, ..., 10] > 0).reshape(im_height, im_width, 1)
    vehicle_small_overlay_rgba = np.concatenate((vehicle_small_overlay, vehicle_small_overlay / 2, np.zeros(vehicle_small_overlay.shape), vehicle_small_overlay * 0.5), axis=2)
    # background = white
    background_overlay = np.ones(buildings_overlay.shape) - np.maximum.reduce([buildings_overlay, structures_overlay, road_overlay, track_overlay, trees_overlay, crops_overlay, waterway_overlay, standing_water_overlay, vehicle_large_overlay, vehicle_small_overlay])
    background_overlay_rgba = np.concatenate((background_overlay, background_overlay, background_overlay, background_overlay * 0.5), axis=2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(X[ix], interpolation="bilinear")
    ax.imshow(background_overlay_rgba, interpolation="bilinear")
    ax.imshow(buildings_overlay_rgba, interpolation="bilinear")
    ax.imshow(structures_overlay_rgba, interpolation="bilinear")
    ax.imshow(road_overlay_rgba, interpolation="bilinear")
    ax.imshow(track_overlay_rgba, interpolation="bilinear")
    ax.imshow(trees_overlay_rgba, interpolation="bilinear")
    ax.imshow(crops_overlay_rgba, interpolation="bilinear")
    ax.imshow(waterway_overlay_rgba, interpolation="bilinear")
    ax.imshow(standing_water_overlay_rgba, interpolation="bilinear")
    ax.imshow(vehicle_large_overlay_rgba, interpolation="bilinear")
    ax.imshow(vehicle_small_overlay_rgba, interpolation="bilinear")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# Check if training data looks alright
plot_sample(X_train, y_train, preds_train, preds_train_t, ix=None)

# Check if validation data looks alright
plot_sample(X_test, y_test, preds_val, preds_val_t, ix=None)
