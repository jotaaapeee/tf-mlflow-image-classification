import mlflow
import mlflow.tensorflow
import tensorflow as tf
from pyspark.sql import SparkSession
from data_loader import load_dataset
from model import build_model
from utils import prepare_data

mlflow.set_experiment("tf-mlflow-image-classification")

def main():
    spark = SparkSession.builder.appName("TF_MLflow_Image_Classification").getOrCreate()

    data_path = "data/raw/"

    df = load_dataset(spark, data_path)

    train_ds, test_ds = prepare_data(df)

    model = build_model()

    epochs = 5
    batch_size = 32
    img_size = (128, 128)

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("img_size", img_size)

        history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)

        mlflow.log_metric("loss", history.history["loss"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        mlflow.tensorflow.log_model(model, "model")

    spark.stop()


if __name__ == "__main__":
    main()
