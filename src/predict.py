import argparse
import tensorflow as tf
import mlflow.tensorflow
import os

IMG_SIZE = (128, 128)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return tf.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()

    model_uri = f"runs:/{args.run_id}/model"
    model = mlflow.tensorflow.load_model(model_uri)

    img = load_image(args.image)
    preds = model.predict(img)

    class_id = preds.argmax()
    confidence = preds[0][class_id]

    print(f"Predicted class: {class_id}")
    print(f"Confidence: {confidence:.4f}")
    print("Raw prediction:", preds)


if __name__ == "__main__":
    main()
