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

def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def to_numpy(dataset):
    dataset = dataset.map(normalize_img)

    combined_data = []

    for image_batch, label_batch in train_ds:
        for img, lbl in zip(image_batch.numpy(), label_batch.numpy()):
            combined_data.append((img, lbl))

    combined_data = np.array(combined_data, dtype=object)

    return combined_data

def partition(dataset, num_clients: int):
    dataset = dataset.map(normalize_img)

    combined_data = to_numpy(dataset)

    #   num_batches = len(combined_data)

    #   batches_per_client = num_batches // num_clients

    sub_datasets = np.array_split(combined_data, num_clients)

    #   for i in range(num_clients):
    #       start = i * batches_per_client
    #       end = (i + 1) * batches_per_client
    #       if i == num_clients - 1:
    #           end = num_batches
    #       sub_dataset = dataset.skip(start).take(batches_per_client)
    #       sub_dataset = sub_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    #       sub_datasets.append(sub_dataset)
        
    return sub_datasets

def get_dataset(num_clients: int):
    train_loaders = partition(train_ds, num_clients)
    val_loaders = partition(val_ds, num_clients)
    test_loaders = to_numpy(test_ds)

    return train_loaders, val_loaders, test_loaders

train, _, _ = get_dataset(10)