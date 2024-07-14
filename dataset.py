import keras
import tensorflow as tf
import numpy as np

batch_size = 32
img_height = 180
img_width = 180

train_ds = keras.utils.image_dataset_from_directory(
    "./Fruits_Classification/train",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    "./Fruits_Classification/valid",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = keras.utils.image_dataset_from_directory(
    "./Fruits_Classification/test",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

def partition(dataset, num_clients: int):
  dataset = dataset.shuffle(buffer_size=1000, seed=123)

  num_batches = tf.data.experimental.cardinality(dataset).numpy()

  batches_per_client = num_batches // num_clients

  sub_datasets = []

  for i in range(num_clients):
      start = i * batches_per_client
      end = (i + 1) * batches_per_client
      if i == num_clients - 1:
          end = num_batches
      sub_dataset = dataset.skip(start).take(batches_per_client)
      sub_dataset = sub_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
      sub_datasets.append(sub_dataset)
    
  return sub_datasets

def get_dataset(num_clients: int):
    train_loaders = partition(train_ds, num_clients)
    val_loaders = partition(val_ds, num_clients)
    test_loaders = test_ds

    return train_loaders, val_loaders, test_loaders