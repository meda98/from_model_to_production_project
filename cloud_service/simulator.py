import time
import requests
import pandas as pd

# URL of the Cloud Run service (production endpoint)
API_URL = "https://anomaly-api-769310733634.europe-west1.run.app/predict"

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
        # Send POST request to the Cloud Run prediction endpoint
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()  # Raise error for HTTP 4xx/5xx

        response_data = response.json()

        # Print input and model output
        print(f"Row {idx} -> Input: {payload} -> Response: {response_data}")

    except requests.exceptions.RequestException as e:
        print(f"Request error at row {idx}: {e}")

    # Wait 1 second before sending the next "measurement"
    time.sleep(1)