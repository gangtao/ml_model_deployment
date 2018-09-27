import org.apache.spark.ml._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.SparkFiles

spark.sparkContext.addFile("https://s3-us-west-2.amazonaws.com/mlapi-samples/demo/data/input/iris.csv")
val data = spark.read.format("csv").option("header", "true").load(SparkFiles.get("iris.csv"))
// Transform, convert string coloumn to number
val featureDf = data.select(data("sepal_length").cast(DoubleType).as("sepal_length"),
                            data("sepal_width").cast(DoubleType).as("sepal_width"),
                            data("petal_width").cast(DoubleType).as("petal_width"),
                            data("petal_length").cast(DoubleType).as("petal_length"),
                            data("species") )

val pipeline = Pipeline.read.load("classification-pipeline")
val model = PipelineModel.read.load("classification-model")

println(pipeline.getStages.size)

// Make predictions.
val predictions = model.transform(featureDf)

// Select example rows to display.
predictions.select("predictedLabel", "species", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
