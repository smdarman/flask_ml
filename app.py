from sklearn.preprocessing import StandardScaler
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("diab_model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    scaler = StandardScaler()
    data = scaler.fit_transform(features)
    prediction = model.predict(data)
    return render_template("index.html", prediction_text = "The probability of getting diabetes is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)