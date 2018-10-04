# Initialize MLeap libraries before Scikit/Pandas
import mleap.sklearn.preprocessing.data
import mleap.sklearn.pipeline
from mleap.sklearn.ensemble import forest
from mleap.sklearn.preprocessing.data import FeatureExtractor

# Import Scikit Transformer(s)
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
input_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

output_vector_name = 'extracted_features' # Used only for serialization purposes
output_features = [x for x in input_features]

feature_extractor_tf = FeatureExtractor(input_scalars=input_features,
                                        output_vector=output_vector_name,
                                        output_vector_items=output_features)

classification_tf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=2, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                       oob_score=False, random_state=0, verbose=0, warm_start=False)

classification_tf.mlinit(input_features="features", prediction_column='species',feature_names="features")

rf_pipeline = Pipeline([(feature_extractor_tf.name, feature_extractor_tf),
                        (classification_tf.name, classification_tf)])
rf_pipeline.mlinit()
rf_pipeline.fit(data[input_features],data['species'])

rf_pipeline.serialize_to_bundle('./', 'mleap-scikit-rf-pipeline', init=True)
