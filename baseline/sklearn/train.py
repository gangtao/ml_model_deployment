from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

data = load_iris()

X, y = data["data"], data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

print(clf.feature_importances_)

print(classification_report(y_test, clf.predict(
    X_test), target_names=data["target_names"]))

joblib.dump(clf, 'classification.pkl')
