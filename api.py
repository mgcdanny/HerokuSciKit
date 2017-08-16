from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn import tree


def train():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(IRIS.data, IRIS.target)
    return clf


IRIS = load_iris()
CLF = train()


app = Flask(__name__)

@app.route("/")
def home():
    return "Hello World!"

@app.route("/api/predict", methods=["POST"])
def predict():
    '''predict takes an array of 4 float values'''
    data = request.get_json()['data']
    pred = CLF.predict([data])
    return jsonify({"data": pred.tolist()})

