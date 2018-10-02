from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import DoubleType, StructType, StructField
from pyspark.sql import SparkSession

import socket

# set the driver host to local ip address
ip = socket.gethostbyname(socket.gethostname())
SparkContext.setSystemProperty('spark.driver.host', ip)
conf = SparkConf().setAppName(
    'spark-deployment').setMaster('spark://spark-master:7077')
sc = SparkContext(conf=conf)


DEFAULT_HDFS_HOST = 'ad95fe885c37011e8aee806444a30499-1181034928.us-west-2.elb.amazonaws.com'


class PySparkModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        self.hdfs = DEFAULT_HDFS_HOST
        self.cSchema = StructType([StructField("sepal_length", DoubleType()),
                                   StructField("sepal_width", DoubleType()),
                                   StructField("petal_width", DoubleType()),
                                   StructField("petal_length", DoubleType())])
        self.model = None

    def predict(self, X, features_names):
        print(X)
        print(type(X))

        spark=SparkSession.builder.master("spark-master").getOrCreate()
        data=spark.createDataFrame(X.tolist(), schema=self.cSchema)

        print("Covert np array to list")
        print(data)
        print(type(data))
        
        if self.model is None:
            # self.pipeline = Pipeline.read().load("hdfs://{}:9000/tmp/classification-pipeline".format(self.server))
            self.model=PipelineModel.read().load(
                "hdfs://{}:9000/tmp/classification-model".format(self.hdfs))

        predictions=self.model.transform(data)
        labels = predictions.select('predictedLabel').collect()
        return [str(x.predictedLabel) for x in labels]
