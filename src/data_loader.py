from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
import tensorflow as tf

def load_dataset(spark, path):
    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(path)
    )
    return df
