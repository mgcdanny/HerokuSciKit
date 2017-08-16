from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn import tree


def train(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    return clf

# overwrite this method for your use case
# 
def load_data_for_training():
    return load_iris()


X, Y = load_data_for_training()

CLF = train(training_data)


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

