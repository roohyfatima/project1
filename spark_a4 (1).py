from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("FuelPricePrediction").getOrCreate()

# Load data from CSV into a Spark DataFrame
data_path = "s3://projectaug25/Input/all_fuels_data.csv"
data_df = spark.read.csv(data_path, header=True, inferSchema=True)

# Select relevant columns and rename 'close' column to 'label'
selected_data = data_df.select("open", "high", "low", "volume", "close") \
                        .withColumnRenamed("close", "label")

# Data Preprocessing and Feature Engineering
assembler = VectorAssembler(inputCols=["open", "high", "low", "volume"], outputCol="features")
assembled_df = assembler.transform(selected_data)

# Split Data into Training and Testing Sets
train_ratio = 0.8
test_ratio = 1.0 - train_ratio
train_data, test_data = assembled_df.randomSplit([train_ratio, test_ratio], seed=123)

# Build and Train a Machine Learning Model (Random Forest Regressor)
rf_regressor = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=100)
model = rf_regressor.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model's performance
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
output_file_path = "s3://projectaug25/Output/output.txt"
with open(output_file_path, "w") as f:
    f.write(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Stop Spark session
spark.stop()
