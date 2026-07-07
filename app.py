from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("temperature_prediction_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    humidity = float(request.form['humidity'])
    pressure = float(request.form['pressure'])
    wind_speed = float(request.form['wind_speed'])
    visibility = float(request.form['visibility'])
    cloud_cover = float(request.form['cloud_cover'])
    dew_point = float(request.form['dew_point'])

    features = np.array([[humidity,
                          pressure,
                          wind_speed,
                          visibility,
                          cloud_cover,
                          dew_point]])

    prediction = model.predict(features)[0]

    return render_template(
        "result.html",
        prediction=round(prediction,2)
    )


if __name__ == "__main__":
    app.run(debug=True)