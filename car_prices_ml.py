from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Spark session
spark = SparkSession.builder.appName("CarPricesML").enableHiveSupport().getOrCreate()

# Load data from Hive
df = spark.sql("""
    SELECT year, odometer, mmr, sellingprice
    FROM carprices
""")

# Handle nulls
df = df.na.drop()

# Assemble features
assembler = VectorAssembler(inputCols=["year", "odometer", "mmr"], outputCol="features", handleInvalid="skip")
assembled_df = assembler.transform(df).select("features", "sellingprice")

# Split into train/test
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Train Linear Regression
lr = LinearRegression(labelCol="sellingprice")
lr_model = lr.fit(train_data)

# Evaluate
results = lr_model.evaluate(test_data)
print(f"RMSE: {results.rootMeanSquaredError}")
print(f"R2: {results.r2}")

# Write metrics to HBase
data = [
    ('metrics1', 'details:rmse', str(results.rootMeanSquaredError)),
    ('metrics1', 'details:r2', str(results.r2))
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('car_sales')
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Stop Spark
spark.stop()
