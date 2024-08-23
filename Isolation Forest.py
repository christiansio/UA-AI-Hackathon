import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Load the dataset with 'DATETIME READ' parsed as dates
# Modify file path depending on location of dataset
file_path = r'C:\Users\Chris\Documents\UA-AI Hackathon\Split\split_1793_3_HYDROMET_6_ANTIQUE_SANREMIGIO_BUGOBRIDGE.csv'
df_re = pd.read_csv(file_path, parse_dates=['DATETIME READ'])

# Display data types of all columns
print("Data Types:\n", df_re.dtypes)

# Display first few entries of 'DATETIME READ'
print("\nFirst few 'DATETIME READ' entries:\n", df_re['DATETIME READ'].head())

# Check if 'DATETIME READ' is already datetime
if not pd.api.types.is_datetime64_any_dtype(df_re['DATETIME READ']):
    print("\nConverting 'DATETIME READ' to datetime...")
    df_re['DATETIME READ'] = pd.to_datetime(df_re['DATETIME READ'], errors='coerce')
    
    # Verify conversion
    print("After conversion, 'DATETIME READ' dtype:", df_re['DATETIME READ'].dtype)
    print("First few entries after conversion:\n", df_re['DATETIME READ'].head())

    # Drop rows where 'DATETIME READ' couldn't be parsed
    initial_row_count = df_re.shape[0]
    df_re = df_re.dropna(subset=['DATETIME READ'])
    final_row_count = df_re.shape[0]
    dropped_rows = initial_row_count - final_row_count
    if dropped_rows > 0:
        print(f"\nDropped {dropped_rows} rows due to unparsable 'DATETIME READ' entries.")

# Set 'DATETIME READ' as the index
df_re.set_index('DATETIME READ', inplace=True)

# Verify the index type
print("\nIndex Type:", type(df_re.index))
print("Index Dtype:", df_re.index.dtype)

# Ensure the index is a DatetimeIndex
if not isinstance(df_re.index, pd.DatetimeIndex):
    raise TypeError("Index is not a DatetimeIndex. Please check your input data.")

# Convert all relevant columns to numeric, forcing errors to NaN
df_re['RAINFALL AMOUNT (mm)'] = pd.to_numeric(df_re['RAINFALL AMOUNT (mm)'], errors='coerce')
df_re['WATERLEVEL (m)'] = pd.to_numeric(df_re['WATERLEVEL (m)'], errors='coerce')
df_re['WATERLEVEL MSL (m)'] = pd.to_numeric(df_re['WATERLEVEL MSL (m)'], errors='coerce')

# Drop rows where any column has NaN
df_re = df_re.dropna(how='any', axis=0)

# Interpolate to handle any remaining missing data
df_re.interpolate(method='linear', inplace=True)

def resampled_data(df):
    """
    Function: Downsamples (high frequency to low frequency) input data
    Input: Dataframe
    Output: Resampled dataframe + CSV file
    """
    # Resample all required columns
    data = pd.DataFrame()
    data['RAINFALL AMOUNT (mm)'] = df['RAINFALL AMOUNT (mm)'].resample('D').mean().ffill()
    data['WATERLEVEL (m)'] = df['WATERLEVEL (m)'].resample('D').mean().ffill()
    data['WATERLEVEL MSL (m)'] = df['WATERLEVEL MSL (m)'].resample('D').mean().ffill()

    # Add the 'Date' column from the index
    new_data = data.copy()
    new_data['Date'] = new_data.index

    # Save the resampled data to CSV
    new_data.to_csv('Resampled_input.csv', index=False)

    return new_data

# Resample the data
resampled = resampled_data(df_re)

# Apply Isolation Forest model
model = IsolationForest(random_state=0, contamination=0.01)
model.fit(resampled[['RAINFALL AMOUNT (mm)', 'WATERLEVEL (m)', 'WATERLEVEL MSL (m)']])

# Add anomaly scores and labels
resampled['score'] = model.decision_function(resampled[['RAINFALL AMOUNT (mm)', 'WATERLEVEL (m)', 'WATERLEVEL MSL (m)']])
resampled['anomaly_value'] = model.predict(resampled[['RAINFALL AMOUNT (mm)', 'WATERLEVEL (m)', 'WATERLEVEL MSL (m)']])

# Extract outliers
outliers = resampled.loc[resampled['anomaly_value'] == -1]

# Create a copy of the resampled data without anomalies
resampled_no_anomalies = resampled[resampled['anomaly_value'] != -1]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# Plot with anomalies highlighted
axes[0].plot(resampled['Date'], resampled['RAINFALL AMOUNT (mm)'], label='RAINFALL AMOUNT (mm)', alpha=0.5)
axes[0].plot(resampled['Date'], resampled['WATERLEVEL (m)'], label='WATERLEVEL (m)', alpha=0.5)
axes[0].plot(resampled['Date'], resampled['WATERLEVEL MSL (m)'], label='WATERLEVEL MSL (m)', alpha=0.5)
axes[0].scatter(outliers['Date'], [0]*len(outliers), color='red', label='Anomalies', edgecolor='black', s=150, zorder=5, marker='o')
axes[0].set_title('With Anomalies', fontsize=16)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True)

# Plot without anomalies
axes[1].plot(resampled_no_anomalies['Date'], resampled_no_anomalies['RAINFALL AMOUNT (mm)'], label='RAINFALL AMOUNT (mm)', alpha=0.5)
axes[1].plot(resampled_no_anomalies['Date'], resampled_no_anomalies['WATERLEVEL (m)'], label='WATERLEVEL (m)', alpha=0.5)
axes[1].plot(resampled_no_anomalies['Date'], resampled_no_anomalies['WATERLEVEL MSL (m)'], label='WATERLEVEL MSL (m)', alpha=0.5)
axes[1].set_title('Without Anomalies', fontsize=16)
axes[1].set_xlabel('Date')
axes[1].legend()
axes[1].grid(True)

# Show the plots
plt.tight_layout()
plt.show()

# Define the output directory
# Modify depending on where you want the anomaly file to go
output_dir = r'C:\Users\Chris\Documents\Python codes\IF-Anomalies'

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the output file path
output_file = os.path.join(output_dir, 'HYDROMET_6_ANTIQU   E_SANREMIGIO_BUGOBRIDGE.csv')

# Save the anomalies to a CSV file
outliers.to_csv(output_file, index=False)

print(f"Anomalies exported to {output_file}")
