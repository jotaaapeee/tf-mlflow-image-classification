import tensorflow as tf

def build_model(input_shape=(128, 128, 3), num_classes=2):
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
