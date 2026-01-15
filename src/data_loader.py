from pyspark.sql import Row
import os

def load_dataset(spark, base_path):
    rows = []

    for label_name, label_id in [("cat", 0), ("dog", 1)]:
        folder = os.path.join(base_path, label_name)

        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(folder, filename)
                rows.append(Row(path=full_path, label=label_id))

    return spark.createDataFrame(rows)
