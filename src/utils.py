import tensorflow as tf

def decode_tf_image(content):
    img = tf.image.decode_jpeg(content, channels=3)
    img = tf.image.resize(img, (128, 128))
    return img

def prepare_data(df):
    pandas_df = df.toPandas()

    images = []
    labels = []

    for _, row in pandas_df.iterrows():
        img_tensor = decode_tf_image(row["content"])
        images.append(img_tensor)

        label = 0 if "cat" in row["path"] else 1
        labels.append(label)

    images = tf.stack(images)
    labels = tf.convert_to_tensor(labels)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(500).batch(32)

    train_size = int(0.8 * len(images))

    train_ds = dataset.take(train_size)
    test_ds = dataset.skip(train_size)

    return train_ds, test_ds
