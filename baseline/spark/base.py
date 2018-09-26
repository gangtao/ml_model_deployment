
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import DoubleType

from pyspark import SparkFiles

url = "https://s3-us-west-2.amazonaws.com/mlapi-samples/demo/data/input/iris.csv"
spark.sparkContext.addFile(url)

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.csv(SparkFiles.get("iris.csv"), header=True)

data = data.withColumn("sepal_length", data["sepal_length"].cast(DoubleType()))
data = data.withColumn("sepal_width", data["sepal_width"].cast(DoubleType()))
data = data.withColumn("petal_width", data["petal_width"].cast(DoubleType()))
data = data.withColumn("petal_length", data["petal_length"].cast(DoubleType()))


#data.show()
data.printSchema()

assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_width", "petal_length"],
    outputCol="features")

output = assembler.transform(data)

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="species", outputCol="indexedLabel").fit(output)


# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(output)


# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = output.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "species", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only
