import keras

def get_model(num_classes: int):
    base_model = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(180, 180, 3))
    """Constructs a simple model architecture suitable for MNIST."""
    x = keras.layers.Flatten()(base_model.output)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model;