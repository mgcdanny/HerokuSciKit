from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn import tree
app = Flask(__name__)


# overwrite this method for your use case
def train(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    return clf

# overwrite this method for your use case
def load_data_for_training():
    iris = load_iris()
    return iris.data, iris.target

# alternatively, if model is already trained, grab it from the database
def load_model():
    pass

X, Y = load_data_for_training()

CLF = train(X, Y)


@app.route("/")
def home():
    return """
    Iris Prediction Service:

    curl -H "Content-Type: application/json"  -d '{{"data": [1,2,3,1]}}' https://heroku-predict.herokuapp.com/api/predict
    
    {}

    """.format(test_precision)


@app.route("/api/predict", methods=["POST"])
def predict():
    '''predict takes an array of 4 float values'''
    data = request.get_json()['data']
    pred = CLF.predict([data])
    return jsonify({"data": pred.tolist()})

