# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:47:16 2019

@author: Aidan
"""

from collections import defaultdict
import csv
import sys
import os

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff
import pandas as pd

csv.field_size_limit(2147483647)
base_dir = os.getcwd()


def get_scalers():
    h, w = im_size
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def mask_for_polygons(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def show_mask(m):
    tiff.imshow(255 * np.stack([m, m, m]))


args = sys.argv
print(str(args))
if len(args) != 3:
    print("Error: not enough arguments")

train_data = np.zeros((1, 3335, 3335, 3))
train_labels = np.zeros((1, 3335, 3335, 10), dtype=np.short)
i = 0

first_file = pd.read_csv(os.path.join(base_dir, "grid_sizes.csv"))
#second_file = pd.read_csv(os.path.join(base_dir, "train_wkt_v4.csv"))
for im_id in os.listdir(os.path.join(base_dir, "train_geojson_v3/train_geojson_v3"))[1:]:
    if im_id == str(args[1]):
        x_max = y_min = None
        for index, row in first_file.iterrows():
            _im_id, _x, _y = row
            if _im_id == im_id:
                x_max, y_min = float(_x), float(_y)
                break
        train_polygons = []
        for poly_type in range(1, 11):
            for _im_id, _poly_type, _poly in csv.reader(open(os.path.join(base_dir, "train_wkt_v4.csv"))):
                if _im_id == im_id and _poly_type == str(poly_type):
                    train_polygons.append(shapely.wkt.loads(_poly))
                    break

        im_rgb = tiff.imread(os.path.join(base_dir, "three_band/three_band/{}.tif".format(im_id))).transpose([1, 2, 0])
        im_size = im_rgb.shape[:2]

        x_scaler, y_scaler = get_scalers()

        train_polygons_scaled = []
        for poly_type in range(10):
            train_polygons_scaled.append(shapely.affinity.scale(train_polygons[poly_type], xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0)))

        train_masks = np.zeros((3335, 3335, 10))
        for poly_type in range(10):
            train_masks[..., poly_type] = mask_for_polygons(train_polygons_scaled[poly_type])[0:3335, 0:3335]

        train_data[i] = scale_percentile(im_rgb)[0:3335, 0:3335, :]
        train_labels[i] = train_masks
        i += 1

new_train_data = np.zeros((676, 160, 160, 3))
new_train_labels = np.zeros((676, 160, 160, 11), dtype=np.short)

pos = 0
for i in range(26):
    for j in range(26):
        new_train_data[pos] = train_data[0, (i*160-i*33):((i+1)*160-i*33), (j*160-j*33):((j+1)*160-j*33), :]
        new_train_labels[pos, ..., 1:] = train_labels[0, (i*160-i*33):((i+1)*160-i*33), (j*160-j*33):((j+1)*160-j*33), :]
        pos += 1

new_train_labels[..., 0] = 1 - (new_train_labels.any(axis=3)).squeeze()

tiff.imshow(train_data[0])
show_mask(train_labels[0].any(axis=2))
tiff.imshow(new_train_data[0])
show_mask(new_train_labels[0][..., 1:].any(axis=2))

ix = int(args[2])
np.save(os.path.join(base_dir, "data/train_data_" + str(ix) + ".npy"), new_train_data)
np.save(os.path.join(base_dir, "labels/train_labels_" + str(ix) + ".npy"), new_train_labels)
