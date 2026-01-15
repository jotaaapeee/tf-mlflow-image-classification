import tensorflow as tf

def decode_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = img / 255.0
    return img, label


def prepare_data(df, batch_size=32):
    paths = df.select("path").rdd.flatMap(lambda x: x).collect()
    labels = df.select("label").rdd.flatMap(lambda x: x).collect()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds.take(int(0.8 * len(paths))), ds.skip(int(0.8 * len(paths)))
