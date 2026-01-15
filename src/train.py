import mlflow
from sklearn.metrics import classification_report
import mlflow.tensorflow
import tensorflow as tf
from pyspark.sql import SparkSession
from data_loader import load_dataset
from model import build_model
from utils import prepare_data

MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("tf-mlflow-image-classification")

def main():
    with mlflow.start_run(run_name="tf_spark_training"):
        spark = (
            SparkSession.builder
            .appName("TF_MLflow_Image_Classification")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

        data_path = "data/raw/"
        df = load_dataset(spark, data_path)

        train_ds, test_ds = prepare_data(df)

        model = build_model()

        epochs = 15
        batch_size = 32
        img_size = (128, 128)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("img_size", img_size)

        for x, y in train_ds.take(1):
            print(x.shape, y)

        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs
        )

        mlflow.log_metric("loss", history.history["loss"][-1])

        if "val_loss" in history.history:
            mlflow.log_metric("val_loss", history.history["val_loss"][-1])

        if "val_accuracy" in history.history:
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        mlflow.tensorflow.log_model(model, artifact_path="model")

        for epoch, acc in enumerate(history.history["accuracy"]):
            mlflow.log_metric("accuracy", acc, step=epoch)
            
        if "val_accuracy" in history.history:
            for epoch, v_acc in enumerate(history.history["val_accuracy"]):
                mlflow.log_metric("val_accuracy", v_acc, step=epoch)

        spark.stop()


if __name__ == "__main__":
    main()
