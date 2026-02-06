import time
import requests
import pandas as pd

# URL of your locally running Flask API
API_URL = "http://127.0.0.1:5000/predict"

# Load the same dataset used for training
df = pd.read_csv("predictive_maintenance.csv")

# Rename columns to match training and API expectations
df = df.rename(columns={
    "Air temperature [K]": "temperature",
    "Process temperature [K]": "humidity",
    "Rotational speed [rpm]": "sound_volume"
})

# Select only the columns we need for the API
sensor_data = df[["temperature", "humidity", "sound_volume"]]

# Iterate over the rows to simulate a continuous stream
for idx, row in sensor_data.iterrows():
    # Build the JSON payload for the API
    payload = {
        "temperature": float(row["temperature"]),
        "humidity": float(row["humidity"]),
        "sound_volume": float(row["sound_volume"])
    }

    try:
        # Send POST request to the /predict endpoint
        response = requests.post(API_URL, json=payload)
        response_data = response.json()

        # Print input and model output
        print(f"Row {idx} -> Input: {payload} -> Response: {response_data}")

    except Exception as e:
        print(f"Error at row {idx}: {e}")

    # Wait 1 second before sending the next "measurement"
    time.sleep(1)