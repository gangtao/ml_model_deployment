import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{IntegerType, DoubleType}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import org.apache.spark.SparkFiles

spark.sparkContext.addFile("https://s3-us-west-2.amazonaws.com/mlapi-samples/demo/data/input/iris.csv")
val data = spark.read.format("csv").option("header", "true").load(SparkFiles.get("iris.csv"))

//data.show()
//data.printSchema()

// Transform, convert string coloumn to number
val featureDf = data.select(data("sepal_length").cast(DoubleType).as("sepal_length"),
                            data("sepal_width").cast(DoubleType).as("sepal_width"),
                            data("petal_width").cast(DoubleType).as("petal_width"),
                            data("petal_length").cast(DoubleType).as("petal_length"),
                            data("species") )

// assember the features
val assembler = new VectorAssembler()
  .setInputCols(Array("sepal_length", "sepal_width", "petal_width", "petal_length"))
  .setOutputCol("features")
  
val output = assembler.transform(featureDf)

// create lable and features
val labelIndexer = new StringIndexer()
  .setInputCol("species")
  .setOutputCol("indexedLabel")
  .fit(output)

val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(output)
  
// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)


// Make predictions.
val predictions = model.transform(testData)


// Select example rows to display.
predictions.select("predictedLabel", "species", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)

