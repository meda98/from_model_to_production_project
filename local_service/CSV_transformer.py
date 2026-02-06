import pandas as pd

# Read your CSV file
df = pd.read_csv("predictive_maintenance.csv")

# Write to Excel
df.to_excel("predictive_maintenance.xlsx", index=False)