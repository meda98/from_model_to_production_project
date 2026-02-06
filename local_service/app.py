from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the trained model once when the service starts
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract JSON payload from the incoming POST request
    data = request.get_json()

    # Read the sensor values from the JSON body
    temperature = data["temperature"]
    humidity = data["humidity"]
    sound_volume = data["sound_volume"]

    # Use a DataFrame with feature names (matches training)
    X = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "sound_volume": sound_volume
    }])

    # Compute the probability of failure
    prob = model.predict_proba(X)[0, 1]

    # Compute the class prediction (0 = no failure, 1 = failure)
    pred = int(model.predict(X)[0])

    # Return the predictions as a JSON response
    return jsonify({
        "failure_probability": float(prob),
        "prediction": pred
    })

@app.route("/health", methods=["GET"])
def health():
    # Simple endpoint to check if the service is running
    return jsonify({"status": "ok"})

# Start the Flask development server
if __name__ == "__main__":
    app.run(port=5000)