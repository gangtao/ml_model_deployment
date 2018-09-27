from sklearn.externals import joblib

from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

data = load_iris()

X, y = data["data"], data["target"]

clf = joblib.load('classification.pkl')

print(clf.feature_importances_)
print(classification_report(y, clf.predict(
    X), target_names=data["target_names"]))
