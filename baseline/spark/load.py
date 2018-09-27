%pyspark
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

from pyspark import SparkFiles

url = "https://s3-us-west-2.amazonaws.com/mlapi-samples/demo/data/input/iris.csv"
spark.sparkContext.addFile(url)

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.csv(SparkFiles.get("iris.csv"), header=True)

data = data.withColumn("sepal_length", data["sepal_length"].cast(DoubleType()))
data = data.withColumn("sepal_width", data["sepal_width"].cast(DoubleType()))
data = data.withColumn("petal_width", data["petal_width"].cast(DoubleType()))
data = data.withColumn("petal_length", data["petal_length"].cast(DoubleType()))

pipeline = Pipeline.read().load("classification-pipeline")
model = PipelineModel.read().load("classification-model")

# Make predictions.
predictions = model.transform(data)

# Select example rows to display.
predictions.select("predictedLabel", "species", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
