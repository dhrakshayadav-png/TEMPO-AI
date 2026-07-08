from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# ----------------------------------
# Load Machine Learning Model
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "decision_tree_model.pkl"
)

model = joblib.load(model_path)

print("Model Loaded Successfully")


# ----------------------------------
# Home Page
# ----------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------------
# Prediction
# ----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get input values from HTML form
        input_values = []

        for value in request.form.values():
            input_values.append(float(value))

        # Convert to numpy array
        final_input = np.array([input_values])

        # Predict temperature
        prediction = model.predict(final_input)

        result = round(prediction[0], 2)

        return render_template(
            "result.html",
            prediction=result
        )

    except Exception as e:

        return render_template(
            "result.html",
            prediction="Error: " + str(e)
        )


# ----------------------------------
# Run Flask Application
# ----------------------------------

if __name__ == "__main__":
    app.run(debug=True)