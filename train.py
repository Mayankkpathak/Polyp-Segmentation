import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from data import load_data, tf_dataset
from model import build_model


# def iou(y_true, y_pred):
#   def f(y_true, y_pred):
#     intersection = (y_true * y_pred).sum()
#     union = y_true.sum() + y_pred.sum() - intersection
#     x = (intersection + 1e-15)/(union + 1e-15)
#     x = x.astype(np.float32)
#     return x
#   return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    iou = intersection / (union + 1e-15)
    return tf.reduce_mean(iou)



if __name__ == "__main__":
  # Seeding
  np.random.seed(42)
  tf.random.set_seed(42)

  path = "CVC-612"
  (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
  print(len(train_x), len(valid_x), len(test_x))

  # Hyperperameter
  batch = 8
  lr = 1e-4
  epochs = 5

  train_dataset = tf_dataset(train_x, train_y, batch=batch)
  valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

  model = build_model()

  opt = tf.keras.optimizers.Adam(lr)
  metrics = ["acc", Recall(), Precision(), iou]
  model.compile(loss = "binary_crossentropy", optimizer=opt, metrics=metrics)

  callbacks = [
    ModelCheckpoint("files/model.keras"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
    CSVLogger("files/data.csv"),
    TensorBoard(),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
  ]

  train_steps = len(train_x)//batch
  valid_steps = len(valid_x)//batch

  if len(train_x) % batch != 0:
    train_steps += 1
  if len(valid_x) % batch != 0:
    valid_steps += 1

  model.fit(
    train_dataset,
    validation_data  = valid_dataset,
    epochs = epochs,
    steps_per_epoch = train_steps,
    validation_steps = valid_steps,
    callbacks = callbacks,
    shuffle = False
  )